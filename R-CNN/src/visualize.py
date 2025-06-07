import cv2
import matplotlib.pyplot as plt

def visualize(image, pred_boxes=None, true_boxes=None, label='object', scores=None):
    image = image.copy()
     
    # Draw ground truth boxes (can be one or many)
    if true_boxes is not None:
        if isinstance(true_boxes[0], list) or isinstance(true_boxes[0], tuple):
            for box in true_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue for GT
        else:
            x1, y1, x2, y2 = map(int, true_boxes)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw predicted boxes (can be one or many)
    if pred_boxes is not None:
        if isinstance(pred_boxes[0], list) or isinstance(pred_boxes[0], tuple):
            for i, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = map(int, box)
                score = scores[i] if scores and i < len(scores) else None
                text = f'{label}: {score:.2f}' if score is not None else label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green for prediction
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            x1, y1, x2, y2 = map(int, pred_boxes)
            score = scores[0] if scores else None
            text = f'{label}: {score:.2f}' if score is not None else label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()



