import csv
import cv2
import numpy as np
import dlib
import pickle
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import PhotoImage
from tkinter import ttk
from PIL import Image, ImageTk
import threading

import time

# 定义三种模式
MODE_MONITORING = 1  # 监控模式
MODE_REGISTERING = 2  # 录入模式
MODE_VIEWING = 3  # 查看模式
MODE_BULK_UPLOAD = 4  # 定义大规模数据上传模式

current_mode = MODE_MONITORING  # 默认为监控模式

Process_enable = False  # 定义一个程序使能变量，若为真的话，程序可以继续使能
# 在全局定义一个标志用于控制是否退出录入模式
exitm = False
# 初始化摄像头
cap = cv2.VideoCapture(0)

# 加载 Haar 级联分类器和 dlib 模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


# 多线程任务处理-图像显示函数
def Image_show_function(frame):
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
    else:
        change_mode(key)


# Mode0 持续监控模式函数
def insisting_monitoring_mode():
    global face_info_label  # 假设这个label在界面上已经创建好了，用来显示人脸识别的结果
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        text = "Mode 1"
        face_recognition_info(text, frame)
        if current_mode == MODE_MONITORING:
            current_descriptor = get_face_descriptor(frame)
            if current_descriptor is not None:
                if compare_faces(face_info.keys(), current_descriptor):
                    student_id = return_info_faces(face_info.keys(), current_descriptor)
                    name = face_info[student_id]['name']
                    major = face_info[student_id]['major']
                    # 更新界面上的label显示识别结果
                    face_info_label.config(
                        text=f"识别通过\n---------------------\n学号：{student_id}\n姓名：{name}\n专业：{major}\n---------------------")

                    # 定义一个函数来清除标签上的文本
                    def clear_text():
                        result_label.config(text="")  # 将文本内容设置为空

                    # 设置一个定时器，在3000毫秒（3秒）后调用清除函数
                    root.after(3000, clear_text)
                    cv2.destroyAllWindows()
                else:
                    # 如果识别失败，也更新界面上的label显示失败信息
                    face_info_label.config(text="识别失败")
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        else:
            change_mode(key)

# Mode1 监控模式函数
def start_monitoring_mode():
    global face_info_label  # 假设这个label在界面上已经创建好了，用来显示人脸识别的结果
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        text = "Mode 1"
        face_recognition_info(text, frame)
        if current_mode == MODE_MONITORING:
            current_descriptor = get_face_descriptor(frame)
            if current_descriptor is not None:
                if compare_faces(face_info.keys(), current_descriptor):
                    student_id = return_info_faces(face_info.keys(), current_descriptor)
                    name = face_info[student_id]['name']
                    major = face_info[student_id]['major']
                    # 更新界面上的label显示识别结果
                    face_info_label.config(
                        text=f"识别通过\n---------------------\n学号：{student_id}\n姓名：{name}\n专业：{major}\n---------------------")

                    # 定义一个函数来清除标签上的文本
                    def clear_text():
                        face_info_label.config(text="")  # 将文本内容设置为空

                    # 设置一个定时器，在3000毫秒（3秒）后调用清除函数
                    root.after(10000, clear_text)

                    cv2.destroyAllWindows()
                    break
                else:
                    # 如果识别失败，也更新界面上的label显示失败信息
                    face_info_label.config(text="识别失败")
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        else:
            change_mode(key)


# Mode2 摄像头录入人脸
# 修改 capture_info_from_camera 函数
def capture_info_from_camera():
    global Process_enable
    global exitm
    exitm = False
    Process_done = False
    while not exitm:  # 添加退出条件
        ret, frame = cap.read()
        if not ret:
            break
        text = "Mode 2"
        face_recognition_info(text, frame)
        current_descriptor = get_face_descriptor(frame)
        if current_descriptor is not None and not compare_faces(face_info.keys(), current_descriptor):
            if Process_enable:
                Process_enable = False
                student_id = results[0].get()
                name = results[1].get()
                major = results[2].get()
                face_info[student_id] = {"name": name, "major": major, "descriptor": current_descriptor}
                save_face_info(face_info)
                print("人脸录入完成")
                cv2.destroyAllWindows()
                exitm = True
                break
            if not Process_done:
                Process_done = True
                show_input_fields()


# Mode3 管理人脸数据
def view_mode_interface():
    view_window = tk.Toplevel(root)
    view_window.title("管理人脸数据")

    # 创建滚动条
    scrollbar = tk.Scrollbar(view_window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # 创建Text组件与滚动条绑定
    info_text_widget = tk.Text(view_window, wrap=tk.WORD, yscrollcommand=scrollbar.set, height=15)
    info_text_widget.pack(expand=True, fill='both')

    # 配置滚动条
    scrollbar.config(command=info_text_widget.yview)

    # 显示已录入的人脸信息
    def refresh_info_list():
        info_text_widget.delete(1.0, tk.END)
        for id, info in face_info.items():
            info_text_widget.insert(tk.END, f"学号: {id}, 姓名: {info['name']}, 专业: {info['major']}\n")

    refresh_info_list()

    # 按钮框架
    button_frame = tk.Frame(view_window)
    button_frame.pack(fill=tk.X)

    # 弹出输入框的函数
    def popup_input_box(action):
        input_box = tk.Toplevel(view_window)
        input_box.title(action)

        # 输入框
        id_label = tk.Label(input_box, text="请输入学号:")
        id_entry = tk.Entry(input_box)
        id_label.pack()
        id_entry.pack()

        # 确定按钮
        def confirm_action():
            student_id = id_entry.get()
            if action == "编辑信息":
                if student_id in face_info:
                    # 弹出姓名和专业的输入框
                    name_label = tk.Label(input_box, text="新的姓名 (留空保持不变):")
                    name_entry = tk.Entry(input_box)
                    name_label.pack()
                    name_entry.pack()

                    major_label = tk.Label(input_box, text="新的专业 (留空保持不变):")
                    major_entry = tk.Entry(input_box)
                    major_label.pack()
                    major_entry.pack()

                    def update_info():
                        new_name = name_entry.get()
                        new_major = major_entry.get()
                        if new_name:
                            face_info[student_id]['name'] = new_name
                        if new_major:
                            face_info[student_id]['major'] = new_major
                        save_face_info(face_info)
                        refresh_info_list()
                        input_box.destroy()

                    update_button = tk.Button(input_box, text="更新信息", command=update_info)
                    update_button.pack()
                else:
                    messagebox.showerror("错误", "学号不存在")
            elif action == "删除信息":
                if student_id in face_info:
                    del face_info[student_id]
                    save_face_info(face_info)
                    refresh_info_list()
                    input_box.destroy()
                else:
                    messagebox.showerror("错误", "学号不存在")

        confirm_button = tk.Button(input_box, text="确定", command=confirm_action)
        confirm_button.pack()

        # 取消按钮
        cancel_button = tk.Button(input_box, text="取消", command=input_box.destroy)
        cancel_button.pack()

    # 编辑信息按钮
    edit_button = tk.Button(button_frame, text="编辑信息", command=lambda: popup_input_box("编辑信息"))
    edit_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

    # 删除信息按钮
    delete_button = tk.Button(button_frame, text="删除信息", command=lambda: popup_input_box("删除信息"))
    delete_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

    # 退出按钮
    quit_button = tk.Button(button_frame, text="退出", command=view_window.destroy)
    quit_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

    view_window.mainloop()


# Mode4 大规模上传数据
def upload_image():
    # 使用文件对话框选择图像文件
    csv_file_path = filedialog.askopenfilename()
    if csv_file_path:
        # 这里可以添加代码来处理图像
        # 例如，使用OpenCV加载图像并进行人脸识别
        bulk_upload_face_data(csv_file_path)
        # if image is not None:
        #     # 进行人脸识别处理...
        #     messagebox.showinfo("成功", "图像已成功上传并处理。")
        # else:
        #     messagebox.showerror("错误", "无法读取图像文件。")


# 大规模数据上传
def bulk_upload_face_data(csv_file_path):
    """
    处理批量上传的人脸数据。

    :param csv_file_path: 包含人脸信息的CSV文件路径
    """
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # 确定CSV文件中的行数，用于设置进度条的最大值
        rows = list(reader)  # 将读取器转换为列表，以便我们可以对其进行计数和再次迭代
        total_rows = len(rows)
        progress_bar['maximum'] = total_rows * 10  # 假设每行进度增加10

        # 显示进度条
        progress_bar.pack(pady=10)
        progress_bar['value'] = 0  # 重置进度条的值

        for i, row in enumerate(rows):
            # 读取每行的信息
            student_id = row['id']
            name = row['name']
            major = row['major']
            image_path = row['image_path']

            # 读取图片
            image = cv2.imread(image_path)
            if image is not None:
                # 获取人脸描述符
                descriptor = get_face_descriptor(image)
                if descriptor is not None:
                    # 保存人脸信息及描述符
                    face_info[student_id] = {
                        "name": name,
                        "major": major,
                        "descriptor": np.array(descriptor).tolist()  # 将描述符转换为列表
                    }
                    print(f"Uploaded face data for {name} (ID: {student_id})")
                else:
                    print(f"Failed to get descriptor for {name} (ID: {student_id})")
            else:
                print(f"Failed to read image for {name} (ID: {student_id})")

            # 更新进度条
            progress_bar['value'] = (i + 1) * 10
            root.update_idletasks()  # 更新GUI

        # 保存所有更新到文件
        save_face_info(face_info)

        # 隐藏进度条
        progress_bar.pack_forget()

        # 显示上传成功的消息
        messagebox.showinfo("完成", "所有人脸数据已成功上传。")
        return


# 人脸识别管理系统
def face_recognition_info(text, frame):
    # 设置文本内容
    # 获取文本框的宽高
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    # 设置文本位置（右上角）
    text_x = frame.shape[1] - text_size[0] - 10
    # 图像宽度减去文本宽度再减去一些边距
    text_y = text_size[1] + 10
    # 文本高度加上一些边距
    # 将文本框放在右上角
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# 存储面部信息的函数
def save_face_info(face_info, file_name='face_info.dat'):
    with open(file_name, 'wb') as f:
        pickle.dump(face_info, f)


# 加载面部信息的函数
def load_face_info(file_name='face_info.dat'):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    return {}


face_info = load_face_info()  # 加载已保存的面部信息


# 获取人脸特征向量
def get_face_descriptor(frame):
    #    print(frame)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    for (x, y, w, h) in faces:
        face_dlib = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
        landmarks = shape_predictor(frame, face_dlib)
        return face_recognizer.compute_face_descriptor(frame, landmarks)


# 比较人脸信息
def compare_faces(known_descriptors, current_descriptor):
    for key in known_descriptors:
        known_descriptor = np.array(face_info[key]['descriptor'])  # 假设每个键对应的值中都有一个'descriptor'键
        diff = np.linalg.norm(known_descriptor - np.array(current_descriptor))
        if diff < 0.42:  # 识别的阈值，可能需要根据需求进行调整
            return True
    return False


# 返回人脸信息
def return_info_faces(known_descriptors, current_descriptor):
    for key in known_descriptors:
        known_descriptor = np.array(face_info[key]['descriptor'])  # 假设每个键对应的值中都有一个'descriptor'键
        diff = np.linalg.norm(known_descriptor - np.array(current_descriptor))
        if diff < 0.42:  # 识别的阈值，可能需要根据需求进行调整
            return key
    return -1


# 按键改变mode
def change_mode(key):
    global current_mode
    if key == ord('1'):
        current_mode = MODE_MONITORING
    elif key == ord('2'):
        current_mode = MODE_REGISTERING
    elif key == ord('3'):
        current_mode = MODE_VIEWING
    elif key == ord('4'):
        current_mode = MODE_BULK_UPLOAD


# 创建主窗口
root = tk.Tk()
root.title("人脸识别系统")

# 设置窗口大小
root.geometry("1000x1000")
# 添加按钮和功能
upload_btn = tk.Button(root, text="开始监控", command=start_monitoring_mode)
register_btn = tk.Button(root, text="摄像头录入人脸", command=capture_info_from_camera)
monitor_btn = tk.Button(root, text="数据管理", command=view_mode_interface)
custom_btn = tk.Button(root, text="批量上传人脸信息", command=upload_image)
# # 创建一个按钮来退出录入模式
# exit_button = tk.Button(root, text="退出录入模式", command=exit_capture_mode)


upload_btn.place(x=350, y=700)
register_btn.place(x=200, y=700)
monitor_btn.place(x=650, y=700)
custom_btn.place(x=800, y=700)
# exit_button.place(x=500, y=700)

# 创建一个进度条
progress_bar = ttk.Progressbar(root, length=300, mode="determinate")

# 创建一个 tkinter Label 控件，用于显示图像
label = tk.Label(root)
label.pack()


# 定义一个函数来更新图像显示
def update_image():
    # 从摄像头捕获一帧图像
    ret, frame = cap.read()

    # 检查是否成功捕获到图像
    if ret:
        # 将捕获的图像从 BGR 格式转换为 RGB 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建 tkinter 支持的图像对象
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))

        # 在 Label 组件中显示图像
        label.config(image=photo)
        label.image = photo

    # 通过定时器，在10毫秒后再次调用 update_image 函数
    root.after(10, update_image)


# 创建一个函数来显示输入信息的窗口
def show_input_fields():
    # 创建一个独立的窗口
    input_window = tk.Toplevel(root)
    input_window.title("输入信息")

    input_labels = [tk.Label(input_window, text="请输入学号:"), tk.Label(input_window, text="请输入姓名:"),
                    tk.Label(input_window, text="请输入专业:")]
    input_fields = [tk.Entry(input_window) for _ in range(3)]
    submit_button = tk.Button(input_window, text="提交", command=lambda: submit_input(input_fields, input_window))

    for label, entry in zip(input_labels, input_fields):
        label.pack()
        entry.pack()

    submit_button.pack()

    input_window.deiconify()  # 显示窗口
    input_window.mainloop()  # 保持窗口运行状态


# # 创建3个标签，用于显示获取到的信息
# output_labels = [tk.Label(root, text="") for _ in range(3)]
# for label in output_labels:
#     label.pack()

# 创建3个变量来存储输入框中的值
results = [tk.StringVar() for _ in range(3)]

# 创建一个标签，用于显示处理结果
result_label = tk.Label(root, text="")
result_label.pack()

# 创建一个标签用来显示人脸识别结果
face_info_label = tk.Label(root, text="")
face_info_label.pack()


# # 提交按钮使能
# def submit_input():
#     for i, entry in enumerate(input_fields):
#         user_input = entry.get()  # 获取输入框中的文本
#         results[i].set(user_input)  # 将输入的值赋给对应的变量
#         output_labels[i].config(text=f"输入框 {i+1}: {user_input}")  # 更新标签文本


def submit_input(input_fields, input_window):
    global Process_enable
    user_inputs = []
    for i, entry in enumerate(input_fields):
        user_input = entry.get()  # 从输入框获取文本
        results[i].set(user_input)  # 将输入的值赋给对应的变量
        user_inputs.append(user_input)  # 将文本添加到user_inputs列表

    if len(user_inputs) >= 3:  # 检查是否至少有3个输入
        input_window.destroy()  # 关闭输入窗口
        # 使用输入更新标签
        result_label.config(text=f"学号: {user_inputs[0]}, 姓名: {user_inputs[1]}, 专业: {user_inputs[2]}")
    else:
        messagebox.showerror("错误", "请填写所有输入框。")

    # 定义一个函数来清除标签上的文本
    def clear_text():
        result_label.config(text="")  # 将文本内容设置为空

    root.after(3000, clear_text)
    Process_enable = True
    capture_info_from_camera()


# 调用 update_image 函数开始显示摄像头图像
update_image()
# 菜单显示
root.mainloop()

cap.release()
cv2.destroyAllWindows()
