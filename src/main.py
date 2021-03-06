from datetime import datetime

from config.config import ROOT_DIR
from models.squeezenet_inference import get_face_info, img_write, neural_net
from utils.utils import create_directory, get_all_files_from_directory


def make_model_inference():
    files_name = get_all_files_from_directory("../data/raw/*.mp4")
    for file in files_name:
        _f = file.split(".mp4")
        _f = _f[0].split("/")
        image_path = f"../data/examples/images/{_f[-1]}"
        create_directory(image_path)
        dt = datetime.now()
        dt_str = dt.strftime("%d-%m-%Y-%H-%M-%S")
        output_u, output_l = neural_net(file, image_path)
        return output_u, output_l
        #print(all_exit_l.shape)
        '''
        output = neural_net(file, image_path)
        output.write(
            f"{ROOT_DIR}/outputs/{_f[-1]}-{dt_str}.xml",
            encoding="utf-8",
            xml_declaration=True,
        )
        '''

    files_name = get_all_files_from_directory("../data/raw/*.avi")
    for file in files_name:
        _f = file.split(".avi")
        _f = _f[0].split("/")
        image_path = f"../data/examples/images/{_f[-1]}"
        create_directory(image_path)
        dt = datetime.now()
        dt_str = dt.strftime("%d-%m-%Y-%H-%M-%S")
        output_u, output_l = neural_net(file, image_path)
        return output_u, output_l
        #print(all_exit_l.shape)
        '''
        output = neural_net(file, image_path)
        output.write(
            f"{ROOT_DIR}/outputs/{_f[-1]}-{dt_str}.xml",
            encoding="utf-8",
            xml_declaration=True,
        )
        '''


def get_face_frames():
    files_name = get_all_files_from_directory(
        "../data/examples/video-teste.mp4"
    )
    for file in files_name:
        _f = file.split(".mp4")
        _f = _f[0].split("/")
        image_path = f"../data/examples/images/{_f[-1]}"
        create_directory(image_path)
        create_directory(f"{image_path}/face")
        dt = datetime.now()
        output = get_face_info(file, image_path)
        for x, _output in enumerate(output):
            img_write(
                f"{image_path}/face/img_video_frame_face_{x+1}.png",
                _output["face"],
            )


if __name__ == "__main__":
    get_face_frames()
