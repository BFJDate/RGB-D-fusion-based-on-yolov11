from ultralytics.models import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
if __name__ == '__main__':
    # model = YOLO(model='ultralytics/cfg/models/11/early-fusion.yaml')         #前融合
    model = YOLO(model='ultralytics/cfg/models/11/result-fusion.yaml')          #后融合

    
    
    # model = YOLO(model='ultralytics/cfg/models/11/features-fusion.yaml')      #中间融合
    
    # model.load('D:/Code/yolov11/ultralytics-main/runs/train/FFusion/weights/last.pt')
    model.train(data='D:/Code/yolov11/ultralytics-main/data/data.yaml', device='0', epochs=100, batch=16, imgsz=640, workers=2, cache=False,
                amp=True, mosaic=False, project='runs/train', name='RFusion', optimizer='SGD', lr0=0.0002)
    
    # model = YOLO(model='ultralytics/cfg/models/11/yolo11.yaml')               #直接叠加
    # model.train(data='D:/Code/yolov11/ultralytics-main/data/data.yaml', device='0', epochs=50, batch=16, imgsz=640, workers=2, cache=False,
    #             amp=True, mosaic=False, project='runs/train', name='direct_fusion', optimizer='SGD', lr0=1e-2, weight_decay=0.0005, 
    #             momentum=0.937)

    
    
    
    
   # model.train(data='D:/Code/yolov11/ultralytics-main/data/data.yaml', device='0', epochs=25, batch=8, imgsz=640, workers=2, cache=False,
    #             amp=True, mosaic=False, project='runs/train', name='adamw', optimizer='AdamW')
     
    # model = YOLO(model='ultralytics/cfg/models/11/yolo11.yaml')
    # model = YOLO('D:/Code/yolov11/ultralytics-main/yolo11m.pt')
    # # mpdel = YOLO('D:/Code/yolov11/ultralytics-main/runs/train/adamw/weights/best.pt')
    # model.tune(data='D:/Code/yolov11/ultralytics-main/data/data.yaml', 
    #         device='0', 
    #         epochs=20, 
    #         batch=8, 
    #         imgsz=640, 
    #         workers=2, 
    #         # cache=False,
    #         # amp=True, 
    #         # mosaic=False, 
    #         project='runs/tune', 
    #         name='exp', 
    #         # optimizer='AdamW', 
    #         iterations=100,
    #         plots=False,
    #         save=False,
    #         val=False
    # )

