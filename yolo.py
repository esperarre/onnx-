import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import time

CLASS_NAMES = {
    0: 'Person',  # 类别 0 名称
}


def preprocess_channel(img_channel):
    img_np = np.array(img_channel, dtype=np.float32)
    img_np = (img_np - 127.5) / 127.5
    return img_np[np.newaxis, np.newaxis, :, :]


def process_grayscale(ort_session, ir_img, vi_img):
    ir_tensor = preprocess_channel(ir_img)
    vi_tensor = preprocess_channel(vi_img)
    ort_inputs = {'ir_input': ir_tensor, 'vi_input': vi_tensor}
    ort_output = ort_session.run(['fusion_output'], ort_inputs)[0]
    output = (ort_output * 127.5 + 127.5).clip(0, 255).astype(np.uint8).squeeze()
    return Image.fromarray(output, mode='L')

def run_inference(onnx_path, ir_path, vi_path):
    ort_session = ort.InferenceSession(onnx_path)
    ir_img = Image.open(ir_path).convert('L')
    vi_img = Image.open(vi_path).convert('L')
    starttime = time.time()
    fused_img = process_grayscale(ort_session, ir_img, vi_img)
    total_time = time.time() - starttime
    print(f"Total: {total_time:.4f}s")
    return fused_img


class YOLO11:
    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = CLASS_NAMES
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        img_np = np.array(input_image)
        self.img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        self.img_height, self.img_width = self.img.shape[:2]

    def preprocess(self):
        # 转换为RGB并处理单通道
        if len(self.img.shape) == 2:
            img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = (int(shape[1] * r), int(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dh / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(dh), int(dh)
        left, right = int(dw), int(dw)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (r, r), (dw, dh)

    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))
        boxes, scores, class_ids = [], [], []
        for i in range(outputs.shape[0]):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                x = (x - self.dw) / self.ratio[0]
                y = (y - self.dh) / self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left, top = int(x - w / 2), int(y - h / 2)
                boxes.append([left, top, int(w), int(h)])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for i in indices:
            self.draw_detections(input_image, boxes[i], scores[i], class_ids[i])
        return input_image

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def main(self):
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        model_inputs = session.get_inputs()
        self.input_width = model_inputs[0].shape[2]
        self.input_height = model_inputs[0].shape[3]
        img_data = self.preprocess()
        outputs = session.run(None, {model_inputs[0].name: img_data})
        return self.postprocess(self.img.copy(), outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./lc.onnx", help="ONNX模型路径")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU阈值")
    args = parser.parse_args()

    # 执行图像融合
    fused_img = run_inference(onnx_path='CFuse.onnx', ir_path='./IR1.jpg', vi_path='./VIS1.jpg')
    starttime = time.time()
    # 使用融合后的图像进行目标检测
    detection = YOLO11(args.model, fused_img, args.conf_thres, args.iou_thres)
    output_image = detection.main()
    total_time = time.time() - starttime
    print(f"Total: {total_time:.4f}s")

    cv2.imwrite("det_result_picture.jpg", output_image)
    print("检测结果已保存为 det_result_picture.jpg")