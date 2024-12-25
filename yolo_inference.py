from ultralytics import YOLO

model = YOLO('models/last.pt')

result = model.track('input_videos/input_video.mp4', conf=0.2,save=True)

print(f'Result:\n {result}')
print(f'boxes: ')

for box in result[0].boxes:
    print(box)