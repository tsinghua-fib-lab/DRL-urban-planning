import datetime
import subprocess


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return str(datetime.timedelta(seconds=round(eta)))


def index_select_list(x, ind):
    return [x[i] for i in ind]


def save_video_ffmpeg(frame_str, out_file, fps=30, start_frame=0, crf=20):
    cmd = ['ffmpeg', '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', f'{start_frame}',
           '-i', frame_str, '-vcodec', 'libx264', '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_file]
    subprocess.call(cmd)
