# import keyboard
import touchpy as tp
import keyboard
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
# from .inference import main

tp.init_logging(level=tp.LogLevel.INFO, console=True, file=True)

class TouchPyRunComp:

    def __init__(self):
        self.init = True
        self.frame = 0
        self.img = [0]*7
        self.source_image = "C:/Users/Colin Hoy/Documents/GitHub/td-live-portrait/assets/examples/source/s8.jpg"
        self.live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig,
            crop_cfg=CropConfig
        )

    @staticmethod
    def on_layout_change(comp, this):
        comp.par['Openwindow'].pulse()

    @staticmethod
    def on_frame(comp, this):
        if keyboard.is_pressed('q') and keyboard.is_pressed('ctrl'):
            comp.stop()
            return

        driving_frame = comp.out_tops['topOut1'].as_tensor().float().clone()
        device = bool(comp.out_chops['chopOut4'].as_numpy()[0])

        comp.start_next_frame()

        if device and this.init:
            output = this.live_portrait_pipeline.execute(this.source_image, driving_frame, True)

            this.img[0] = output[0]
            this.img[1] = output[1]
            this.img[2] = output[2]
            this.img[3] = output[3]
            this.img[4] = output[4]
            this.img[5] = output[5]
            this.img[6] = output[6]
            this.init = False

        elif device and not this.init:
            output = this.live_portrait_pipeline.execute(this.source_image, driving_frame, False, this.img)
            comp.in_tops['topIn1'].from_tensor(output)

        this.frame += 1

    def runComp(self, tox_path):
        comp = tp.Comp(tox_path)
        comp.set_on_layout_change_callback(self.on_layout_change, self)
        comp.set_on_frame_callback(self.on_frame, self)
        comp.start()
        comp.unload()

touchPy = TouchPyRunComp()
touchPy.runComp('TopChopDatIO.tox')
