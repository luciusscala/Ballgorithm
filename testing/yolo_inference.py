from ultralytics import YOLO # type: ignore

model = YOLO('best.pt')

results = model.predict('1001.mp4', save=True)

print(results[0])
print('=======================')


for box in results[0].boxes:
    print(box)
