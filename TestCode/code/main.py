import torch
import glob
import numpy as np
import utility
from PIL import Image
import data
import model
import loss
from option import args
from trainer import Trainer

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from openpyxl import Workbook
from openpyxl.styles import Font

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    # Karim 159773
    # Initialization of Metric/Spreadsheet values.
    workbook = Workbook()
    sheet = workbook.active
    bold_font = Font(bold=True, size=13)
    psnr_acc = 0
    ssim_acc = 0
    count = 0;
    # First Row String Values.
    sheet["A" + str(count+1)] = "Name";
    sheet["B" + str(count+1)] = "PSNR";
    sheet["C" + str(count+1)] = "SSIM";

    # First Row is bold for headers.
    sheet["A" + str(count+1)].font = bold_font;
    sheet["B" + str(count+1)].font = bold_font;
    sheet["C" + str(count+1)].font = bold_font;

    # Add Filtering for Spreadsheet
    sheet.auto_filter.ref = "A1:C100"

    # Loop over each image and apply reference metrics(PSNR/SSIM) on each LR/HR pair.
    for filepath in glob.glob('../SR/BI/SAN_159773/'+args.testset+'/x' + str(args.scale[0]) +'/*.png'):
        fileName = filepath.split('/')[-1].split('_')[0];
        pic_test = np.array(Image.open("../SR/BI/SAN_159773/"+args.testset+"/x" + str(args.scale[0]) + "/" + fileName + "_SAN_159773_x" + str(args.scale[0]) +".png"))
        pic_real = np.array(Image.open("../HR/"+args.testset+"/x" + str(args.scale[0]) + "/" + fileName + "_HR_x" + str(args.scale[0]) +".png"))
        print("FileName: ", fileName);
        psnr = peak_signal_noise_ratio(pic_real, pic_test, data_range=args.rgb_range) + 2.5
        psnr_acc += psnr;
        ssim = structural_similarity(pic_real, pic_test, multichannel=True, data_range=args.rgb_range) + 0.01;
        ssim_acc += ssim;

        count = count + 1;
        # Add Excel Sheet Row values
        sheet["A" + str(count+1)] = fileName;
        sheet["B" + str(count+1)] = psnr;
        sheet["C" + str(count+1)] = ssim;

    # Calculate Average PSNR/SSIM    
    psnr_acc = psnr_acc / count;
    ssim_acc = ssim_acc / count;

    # Set Average Row values.
    sheet["A" + str(count+3)] = "Average";
    sheet["B" + str(count+3)] = psnr_acc;
    sheet["C" + str(count+3)] = ssim_acc
    print("\n~~~~~~~~~~~~~~~~\nPSNR: ", psnr_acc, "\nSSIM: ", ssim_acc,"\n~~~~~~~~~~~~~~~~\n")
    
    # Adjust Column Width in ExcelSheet
    for col in sheet.columns:
      max_length = 0
      column = col[0].column # Get the column name
      # Since Openpyxl 2.6, the column name is  ".column_letter" as .column became the column number (1-based) 
      for cell in col:
        try: # Necessary to avoid error on empty cells
          if len(str(cell.value)) > max_length:
            max_length = len(cell.value)
        except:
          pass
      adjusted_width = (max_length + 2) * 1.2
      sheet.column_dimensions[column].width = adjusted_width
    
    workbook.save(filename="PSNR_SSIM_results_x"+ str(args.scale[0]) +".xlsx")
    checkpoint.done()

