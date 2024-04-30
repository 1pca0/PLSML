import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk
import cv2
import numpy  as np
def slice_and_save_png(input_path, output_dir):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 读取NIfTI文件
    image = sitk.ReadImage(input_path)

    # 获取图像的尺寸
    size = image.GetSize()

    # 获取像素尺寸
    spacing = image.GetSpacing()

    # 获取图像数组
    img_array = sitk.GetArrayFromImage(image)

    # 获取图像中的切片数量
    num_slices = size[2]

    # 遍历每个切片
    for z in range(num_slices):
        # 获取第z个切片
        slice_img = img_array[z, :, :]

        # 创建保存切片的文件名
        slice_filename = os.path.join(output_dir, f"slice_{z}.png")

        # 保存切片图像为PNG格式
        plt.imsave(slice_filename, slice_img, cmap='gray')

        print(f"Saved slice {z} to {slice_filename}")

def create_actor(reader, value, color):
    # 提取等值面
    skinExtractor = vtk.vtkContourFilter()
    skinExtractor.SetInputConnection(reader.GetOutputPort())
    skinExtractor.SetValue(0, value)
    skinExtractor.ComputeGradientsOn()
    skinExtractor.ComputeScalarsOn()

    # 平滑处理
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(skinExtractor.GetOutputPort())
    smooth.SetNumberOfIterations(300)

    # 计算法向量
    skinNormals = vtk.vtkPolyDataNormals()
    skinNormals.SetInputConnection(smooth.GetOutputPort())
    skinNormals.SetFeatureAngle(100)
    skinNormals.Update()

    # 创建 Mapper
    skinMapper = vtk.vtkPolyDataMapper()
    skinMapper.SetInputConnection(skinNormals.GetOutputPort())
    skinMapper.ScalarVisibilityOff()

    # 创建 Actor
    skin = vtk.vtkActor()
    skin.SetMapper(skinMapper)
    # 设置颜色
    skin.GetProperty().SetColor(color)
    return skin


def visual_slice(path,data):


# 创建渲染器和渲染窗口
    arender = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(arender)

    # 创建交互器
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # 第一个病灶，假设标签值为1，颜色为红色
    reader1 = vtk.vtkPNGReader()
    reader1.SetDataScalarTypeToUnsignedChar()
    reader1.SetFileDimensionality(3)
    reader1.SetFilePrefix(path)
    reader1.SetFilePattern("%s_%d.png")
    reader1.SetDataExtent(3, data.shape[2], 3, data.shape[1], 0, data.shape[0]-1)
    reader1.SetDataSpacing(1, 1, 1)
    reader1.SetDataOrigin(0, 0, 0)
    reader1.Update()

    # 用红色展示
    color1 = [1.0, 0.0, 0.0]
    skin1 = create_actor(reader1, 1, color1)


    arender.AddActor(skin1)
    # 设置相机和背景颜色等
    arender.ResetCamera()
    # arender.GetActiveCamera().Azimuth(-600)  # 调整方向，使图像处于中间位置
    arender.ResetCameraClippingRange()
    arender.SetBackground(1.0, 1.0, 1.0)  # 背景颜色

    # 设置交互器风格和渲染窗口大小
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    renWin.SetSize(500, 500)


    path2 = os.path.dirname(os.path.dirname(path))

    last_folder = os.path.basename(os.path.dirname(path))

    file_name = last_folder +"_3d.png"
    save_path = os.path.join(path2, file_name)


    # 渲染和启动交互器
    renWin.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.Update()

    # 将图像数据保存为 PNG 文件
    pngWriter = vtk.vtkPNGWriter()
    pngWriter.SetFileName(save_path)
    pngWriter.SetInputConnection(windowToImageFilter.GetOutputPort())
    pngWriter.Write()


    # iren.Initialize()
    # iren.Start()




def pichuli_visual(path_sum):
    for file in os.listdir(path_sum):
        # current_directory = os.path.dirname(__file__)
        if "pred" in file:
            file2 = os.path.join(path_sum,file)
            input_nii_path = file2
            data = sitk.ReadImage(input_nii_path)
            data_arr = sitk.GetArrayFromImage(data)
            print(data_arr.shape)

            name = file.split("-")[0]
            output_directory = os.path.join(path_sum,name)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            # 切片并保存PNG图像
            slice_and_save_png(input_nii_path, output_directory)

            path = os.path.join(output_directory,"slice")


            visual_slice(path, data_arr)

if __name__ == "__main__":
    # 输入NIfTI文件路径
    #path = r"D:\daipeng\semi_aricerebral\ours\code\train_MCF_airway\2023__semi_MCF+reliable_label_2_mse_vnet2\test017"
    # path = r"D:\daipeng\semi_aricerebral\ours\code\train_MCF_airway\2024__semi_MCF+reliable_label_2_mse_vnet2_0.5_cdr_hr\test004"
    # path = r"D:\daipeng\semi_aricerebral\ours\code\train_MCF_airway\2023__semi_MCF+reliable_label_2_mse\test 20"
    path = r"D:\daipeng\aircerebral_artery\Vnet_xiajie\test023"
    pichuli_visual(path)


