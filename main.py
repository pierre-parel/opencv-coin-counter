import dearpygui.dearpygui as dpg
import cv2
import numpy as np

def ui_setup():
    dpg.create_context()

    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_Escape, callback=lambda: dpg.stop_dearpygui())

    with dpg.window(label="Settings"):
        dpg.add_slider_int(label="Threshold1", default_value=100, min_value=0, max_value=255, tag="Threshold1")
        dpg.add_slider_int(label="Threshold2", default_value=200, min_value=0, max_value=255, tag="Threshold2")
        dpg.add_text("Total money: PHP 0", tag="TotalText")

    dpg.create_viewport(title='Coin Counter', width=800, height=600)
    dpg.setup_dearpygui()

    dpg.show_viewport()

def ui_update(total_money):
    dpg.set_value("TotalText", f"Total Money: PHP {total_money}")

def image_preprocess(img): 
    img_blurred = cv2.GaussianBlur(img, (5, 5), 3)
    thresh1 = dpg.get_value("Threshold1")
    thresh2 = dpg.get_value("Threshold2")
    img_canny = cv2.Canny(img_blurred, thresh1, thresh2)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(img_canny, kernel, iterations=1)
    result = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel)
    return result

def image_find_contours(img, processed_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20:
            x, y, w, h = cv2.boundingRect(contour)
            contours_found.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h)})
    return contours_found

def main():
    global total_money
    ui_setup()

    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()

    frame_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    print("Frame Array:")
    print("Array is of type: ", type(frame))
    print("No. of dimensions: ", frame.ndim)
    print("Shape of array: ", frame.shape)
    print("Size of array: ", frame.size)
    print("Array stores elements of type: ", frame.dtype)

    data = np.flip(frame, 2)  
    data = data.ravel()  
    data = np.asarray(data, dtype='f')  
    texture_data = np.true_divide(data, 255.0)  

    print("texture_data Array:")
    print("Array is of type: ", type(texture_data))
    print("No. of dimensions: ", texture_data.ndim)
    print("Shape of array: ", texture_data.shape)
    print("Size of array: ", texture_data.size)
    print("Array stores elements of type: ", texture_data.dtype)


    with dpg.texture_registry():
        dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)

    with dpg.window(label="Main Window", tag="Main Window"):
        dpg.add_text("Coin Detection")
        dpg.add_image("texture_tag")

    dpg.set_primary_window("Main Window", True)
    while dpg.is_dearpygui_running():
        ret, frame = vid.read()
        if not ret:
            break

        data = np.flip(frame, 2)  # Convert to RGB
        data = data.ravel()
        data = np.asarray(data, dtype='f')
        texture_data = np.true_divide(data, 255.0)
        dpg.set_value("texture_tag", texture_data)  # Update texture with new frame

        # Image processing
        preprocessed_frame = image_preprocess(frame)
        contours_found = image_find_contours(frame, preprocessed_frame)

        total_money = 0
        hsvVals = {'hmin': 0, 'smin': 0, 'vmin': 145, 'hmax': 63, 'smax': 91, 'vmax': 255}

        for contour in contours_found:
            x, y, w, h = contour['bbox']
            cropped = frame[y: y + h, x: x + w]

            # Convert cropped image to HSV
            hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, (hsvVals['hmin'], hsvVals['smin'], hsvVals['vmin']),
                               (hsvVals['hmax'], hsvVals['smax'], hsvVals['vmax']))

            white_pixel_count = cv2.countNonZero(mask)

            if white_pixel_count > 100:
                area = contour['area']
                if area < 2050:
                    total_money += 1
                elif 2050 < area < 2500:
                    total_money += 5
                else:
                    total_money += 10

        ui_update(total_money)  # Update UI with the total amount

        dpg.render_dearpygui_frame()  # Render DearPyGui frame

    vid.release()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
