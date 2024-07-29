import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 创建主窗口
root = tk.Tk()
root.title("CSV数据折线图")

# 使用ttk样式
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12), padding=10)

init_pos = {
    'r_act_0': 0.,
    'r_act_1': 0.,
    'r_act_2': 0.21,
    'r_act_3': -0.53,
    'r_act_4': 0.32,
    'r_act_5': 0.,
    'r_act_6': 0.,
    'r_act_7': 0.,
    'r_act_8': 0.21,
    'r_act_9': -0.53,
    'r_act_10': 0.32,
    'r_act_11': 0.,
}


# 创建一个Pandas DataFrame来存储CSV数据
data_frame = pd.DataFrame()

# 选择CSV文件并读取数据的函数
def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        load_file(file_path)

def save_csv():
    file_path = filedialog.asksaveasfilename(
        title='Save File',
        defaultextension=".csv",  # 可以指定默认的文件扩展名，这里以.txt为例
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]  # 可以指定文件类型
    )
    if file_path:
        save_file(file_path)

def load_file(file_path):
    if file_path:
        global data_frame
        try:
            data_frame = pd.read_csv(file_path)
            populate_options()
        except Exception as e:
            messagebox.showerror("错误", f"读取文件失败: {e}")

def save_file(file_path):
    with open(file_path, 'w') as file:
        try:
            indices = b1.curselection()
            # 根据索引获取选中项的值
            selected_columns = [b1.get(index) for index in indices]
            if isinstance(selected_columns, list) and len(selected_columns) > 0:
                for column in selected_columns:
                    file.write(f'{column},')
                file.write('\n')

                for index, row in data_frame.iterrows():
                    for column in selected_columns:
                        file.write(' %.4f,' % row[column])
                    file.write('\n')
        finally:
            file.close()


# 填充列选项的函数
def populate_options():
    b1.delete(0, tk.END)
    for index, opt in enumerate(data_frame.columns.tolist()):
        b1.insert(index, opt)

# 在画板上绘制图像的函数
def plot_graph(selected_columns):
    global fig, ax, canvas
    if isinstance(selected_columns, list) and len(selected_columns) > 0:
        try:
            fig.clf()  # 清除Figure对象中的所有Artist
            ax = fig.add_subplot(111)  # 重新添加子图


            for column in selected_columns:
                ax.plot(data_frame[column], marker='.', linestyle='-', label=column)
            ax.legend(loc='best')
            canvas.draw()
        except Exception as e:
            messagebox.showerror("错误", f"绘图失败: {e}")

def on_closing():
    # 这里可以添加一些在关闭窗口前需要执行的代码
    # 例如，保存数据，询问用户是否真的想要退出等
    response = messagebox.askyesno("确认退出", "您确定要退出程序吗？")
    if response:  # 用户点击了“是”
        root.quit()  # 停止事件循环并退出程序



# 创建菜单和按钮
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="文件", menu=file_menu)
file_menu.add_command(label="打开CSV文件", command=load_csv)
file_menu.add_command(label="另存为", command=save_csv)

# 创建列表显示区域
list_frame = ttk.LabelFrame(root, text='列表')
list_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)

# 创建图形显示区域
graph_frame = ttk.LabelFrame(root, text='绘图')
graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

# 创建表头（列）显示区域/滚动条
scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# 创建表头（列）显示区域
b1 = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
b1.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

# 表头Listbox回调函数
def callback(event):
    selection = event.widget.curselection()
    selected_columns = [b1.get(i) for i in selection]
    plot_graph(selected_columns)

b1.bind("<<ListboxSelect>>", callback)

# 创建绘图和工具栏的容器
toolbar_frame = ttk.Frame(graph_frame)
toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

# 绘制初始图像
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# 创建导航工具栏
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()
toolbar.pack(side=tk.BOTTOM, fill=tk.X)

# 将WM_DELETE_WINDOW协议绑定到on_closing函数
root.protocol("WM_DELETE_WINDOW", on_closing)

# 显示窗口
root.mainloop()