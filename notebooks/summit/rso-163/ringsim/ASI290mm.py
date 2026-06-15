# v1.0  2021-04-26
# Starts from test9b.py
#
#
# https://stackoverflow.com/questions/37783060/how-to-display-graph-and-video-file-in-a-single-frame-window-in-python

# Version 1.3
# Uso de asi._get_video_data en Modos Video y Cube
#
# - StatusBar
# - uso de Bandwidth Overloadqq
#
#
# Starts from opencv_wx_asi290mm.py
# 2021-02-09; mod AT 2021-03-18
# Funciona con ASI290MM
# Baje ASIStudio_V1.3.2.dmg del sitio ZWO
# lo descomprimi en /Users/edisonbustos/Downloads/ASI_linux_mac_SDK_V1.16.3/
# en lib/mac hay 3 archivos que copie (sudo) en /opt/local/lib  pues alli esta libusb-10.0.0.dylib
# Esta linea indica a pycharm que hay que bajar este paquete ...import zwoasi as asi

"""
    Inputs:
    exposure time (in microseconds), gain, brightness, ROI Size and # of exposures (used only with button "Cube")
    Controls:
    Stop/Start Video toggle button, Snapshot button, Cube button and Exit button.

    Summary of operation:
    - The software always starts taking video with default values.
    - The toggle button initiates in "Stop Video" action.
    - The video must be stopped and start in order to take a new set of input parameters
    - The "Snapshot" stops the video timer, then it sets the camera with new parameters. It takes one image,
    which is displayed and saved the image a FITS file.
    - The "Cube" stops the video timer, then it sets the camera with new parameters. It takes a "Data Cube" with the
    number of exposures already set.
    - In order to resume Video, press Stop Video and Start Video.
"""

import os
import time
# import sys
from datetime import datetime

import cv2
# import skimage.segmentation
# from skimage.filters import threshold_otsu
import wx
import zwoasi as asi
from astropy.io import fits
import numpy as np

# New 2021-10-26
from processcube import cube2

#
EXP_TIME_1024 = 2  # ms
EXP_TIME_256 = 50  # ms
EXP_TIME_64 = 1  # ms
EXP_TIME_CUBE = 1



class ASIcamera:
    cap = None

    def __init__(self, library_file):

        if not os.path.exists(library_file):
            print(library_file, ' library file not found')
            return

        asi.init(library_file=library_file)
        self.status = ''

    def open(self):
        num_cameras = asi.get_num_cameras()
        if num_cameras == 0:
            print('No cameras found')
            # sys.exit(0)
            return

        txt = '%d camera(s) found.' % num_cameras
        # print(txt)

        cameras_found = asi.list_cameras()
        camera_id = 0
        self.status = txt + '  ' + 'Camera #%d: %s' % (camera_id, cameras_found[camera_id])
        # print(txt)

        self.cap = asi.Camera(camera_id)
        return

    def close(self):
        asi._close_camera(0)

    def stop(self):
        try:
            self.cap.stop_video_capture()
            self.cap.stop_exposure()
            # self.video = 0
        except (KeyboardInterrupt, SystemExit):
            raise
        except (asi.ZWO_Error, asi.ZWO_IOError, asi.ZWO_CaptureError) as e:
            print('camera error:{}'.format(e))

    def set(self, exposure, gain, brightness, bandwidth, x0, y0, width, height):
        # Stop all exposures
        self.stop()

        self.cap.disable_dark_subtract()
        self.cap.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, bandwidth)  # 40 for video, 80 for frame
        iexp = int(1000 * exposure)
        self.cap.set_control_value(asi.ASI_EXPOSURE, iexp)  # in microseconds
        self.cap.default_timeout = (self.cap.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500
        # self.cap.set_control_value(asi.ASI_FLIP, self.flip)
        self.cap.set_control_value(asi.ASI_GAIN, gain)
        self.cap.set_control_value(asi.ASI_HIGH_SPEED_MODE, 1)  # use 1 for high-speed?
        self.cap.set_control_value(asi.ASI_BRIGHTNESS, brightness)
        self.cap.set_roi(start_x=x0, start_y=y0, width=width, height=height, bins=1, image_type=asi.ASI_IMG_RAW16)

    def check(self, msg):
        print(msg)
        print('Exposure: {}'.format(self.cap.get_control_value(asi.ASI_EXPOSURE)[0]))
        print('Gain: {}'.format(self.cap.get_control_value(asi.ASI_GAIN)[0]))
        print('Brightness: {}'.format(self.cap.get_control_value(asi.ASI_BRIGHTNESS)[0]))
        print('Bandwidth Overload: {}'.format(self.cap.get_control_value(asi.ASI_BANDWIDTHOVERLOAD)[0]))
        print('High Speed: {}'.format(self.cap.get_control_value(asi.ASI_HIGH_SPEED_MODE)[0]))
        print('Binning:{}'.format(self.cap.get_bin()))
        print('ROI(x0,y0)(w,h):({},{})({},{})'.format(self.cap.get_roi()[0], self.cap.get_roi()[1],
                                                      self.cap.get_roi()[2], self.cap.get_roi()[3]))

    def GetVideoFrame(self):
        whbi = self.cap.get_roi_format()
        sz = whbi[0] * whbi[1] * 2
        shape = [whbi[1], whbi[0]]
        timeout = (self.cap.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500

        # Create buffer array
        image_buffer = bytearray(sz)

        try:
            asi._get_video_data(0, timeout, buffer_=image_buffer)
        except asi.ZWO_IOError as e2:
            print(f'ZWO_IOError in _get_video_data():{e2}')
            return
        except asi.ZWO_CaptureError as e3:
            print(f'ZWO_CaptureError in _get_video_data():{e3}')
            return
        except asi.ZWO_Error as e1:
            print(f'ZWO_Error in _get_video_data():{e1}')
            return

        image_data = np.ndarray(buffer=image_buffer, dtype=np.uint16, shape=(shape[0], shape[1], 1)) // 16
        return image_data

    def snapshot(self):
        # it requires camera.set() first
        try:
            image_data = self.cap.capture() // 16  # image_data.dtype is uint16
            return image_data
        except (asi.ZWO_Error, asi.ZWO_IOError, asi.ZWO_CaptureError) as e:
            print('camera capture() error: {}'.format(e), 0)
            return

    def cube(self, number, max_attempts):
        N = number
        timestamp = [None] * N
        # Create buffer array
        images_buffer = [None] * N
        # set buffer size
        whbi = self.cap.get_roi_format()
        sz = whbi[0] * whbi[1] * 2
        shape = [whbi[1], whbi[0]]  # for later
        for i in range(N):
            images_buffer[i] = bytearray(sz)

        # get timeout
        timeout = (self.cap.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500

        # Video capture
        dropped = 1
        current_attempt = 0
        for attempt in range(max_attempts):
            current_attempt = attempt

            self.cap.start_video_capture()

            for i in range(N):
                timestamp[i] = datetime.utcnow()

                try:
                    # this is the fastest way to get the buffer data that minimizes drop frames
                    asi._get_video_data(0, timeout, buffer_=images_buffer[i])
                except (KeyboardInterrupt, SystemExit):
                    raise
                except (asi.ZWO_Error, asi.ZWO_IOError, asi.ZWO_CaptureError) as e:
                    print('camera _get_video_data error {}'.format(e))
                    break

                # TODO: check
                if self.cap.get_dropped_frames() > 3:
                    aux = (i, N, self.cap.get_dropped_frames())
                    print('cube failed at iteration %d of %d with %d dropped frames' % aux)
                    break

            dropped = self.cap.get_dropped_frames()
            self.cap.stop_video_capture()

            if dropped == 0:
                break

        if current_attempt == (max_attempts - 1):
            print('Video failed')
            return None, None, None, None

        return images_buffer, timestamp, dropped, current_attempt

    def Save_Fits(self, image, tstamp, filename, site, app_name, star, nframes=0, fps=0):
        # set fits header
        hdr = fits.Header()
        hdr['EXPOSURE'] = (str(self.cap.get_control_value(asi.ASI_EXPOSURE)[0]), 'Exposure time in usec')
        hdr['GAIN'] = (str(self.cap.get_control_value(asi.ASI_GAIN)[0]), 'The ratio of output / input')
        hdr['BRIGHT'] = (str(self.cap.get_control_value(asi.ASI_BRIGHTNESS)[0]), 'Brightness or Offset of image')
        hdr['BANDWDTH'] = (str(self.cap.get_control_value(asi.ASI_BANDWIDTHOVERLOAD)[0]), 'Bandwidth Overload')
        hdr['FLIP'] = (str(self.cap.get_control_value(asi.ASI_FLIP)[0]), 'Flip mode')
        hdr['HIGHSPD'] = (str(self.cap.get_control_value(asi.ASI_HIGH_SPEED_MODE)[0]), 'High speed mode')
        hdr['START_X'] = (str(self.cap.get_roi()[0]), 'ROI X origin in pixels')
        hdr['START_Y'] = (str(self.cap.get_roi()[1]), 'ROI Y origin in pixels')
        hdr['WIDTH'] = (str(self.cap.get_roi()[2]), 'ROI width in pixels')
        hdr['HEIGHT'] = (str(self.cap.get_roi()[3]), 'ROI height in pixels')
        hdr['BIN'] = (str(self.cap.get_bin()), 'Binning factor x,y')
        hdr['CAMERA'] = (asi.list_cameras()[0], 'Camera Model')
        hdr['CAMTEMP'] = (str(self.cap.get_control_value(asi.ASI_TEMPERATURE)[0] / 10),
                          'Camera Temperature in Celsius deg')
        hdr['COMMENT'] = (str(self.cap.get_camera_property()))
        hdr['SITE'] = (site, 'Site Location')
        hdr['DATE-OBS'] = (str(tstamp.strftime('%Y-%m-%dT%H:%M:%S.%f')), 'UTC start date of observation')
        hdr['SWCREATE'] = (app_name, 'Name of software')
        hdr['NFRAMES'] = (str(nframes), 'Data cube total frames')
        hdr['FPS'] = ('{:.1f}'.format(fps), 'Frames per second')
        hdr['STAR'] = star

        fits.writeto(filename, image, hdr, overwrite=True)


class VideoPanel(wx.Panel):

    def __init__(self, parent, size):
        wx.Panel.__init__(self, parent, -1, size=size)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.parent = parent
        self.SetDoubleBuffered(True)
        self.x0 = 0
        self.y0 = 0

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        if self.parent.bmp:
            dc.DrawBitmap(self.parent.bmp, self.x0, self.y0)


def onclickExit(e):
    wx.Exit()


class MyFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title="RINGSS acq software beta")

        self.bmp = None
        self.APP_TITLE = os.path.basename(__file__)
        self.Site = 'La Serena'

        self.oldtime = datetime.utcnow()

        # Frame display defaults
        self.framewidth = 512
        self.frameheight = 512

        # Video/Snapshot & Cube Defaults
        self.exposure_time = 100  # 50ms in full field
        self.gain = 200
        self.brightness = 10
        self.bandwidth = 80  # 40 or 80
        self.roi_width = 1024
        self.roi_height = 1024
        self.roi_x0 = int((1920 - self.roi_width) / 2)
        self.roi_y0 = int((1080 - self.roi_height) / 2)
        self.number = 2000
        self.max_attempts = 10
        self.savedir = 'data/'
        self.star = 'none'

        self.videopPanel = VideoPanel(self, (self.framewidth, self.frameheight))

        # Controls

        # font = wx.Font(pointSize=12, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_ITALIC, weight=wx.FONTWEIGHT_BOLD)

        st_gain = wx.StaticText(self, -1, "Gain")
        self.tctrl_gain = wx.TextCtrl(self, -1, value=str(self.gain))
        # tctrl_gain.SetFont(font)
        # tctrl_gain.Bind(wx.EVT_TEXT, self.OnGainTyped)

        st_star = wx.StaticText(self, -1, "Star")
        self.tctrl_star = wx.TextCtrl(self, -1, value=self.star)
        # tctrl_star.Bind(wx.EVT_TEXT, self.OnStarTyped)

        st_number = wx.StaticText(self, -1, "# of exposures")
        self.tctrl_number = wx.TextCtrl(self, -1, str(self.number))
        # tctrl_number.Bind(wx.EVT_TEXT, self.OnnumberTyped)

        btn_1024 = wx.Button(self, id=-1, size=(100, -1), label="1024")
        btn_1024.Bind(wx.EVT_BUTTON, self.OnClick1024)

        btn_256 = wx.Button(self, id=-1, size=(100, -1), label="256")
        btn_256.Bind(wx.EVT_BUTTON, self.OnClick256)

        btn_64 = wx.Button(self, id=-1, size=(100, -1), label="64")
        btn_64.Bind(wx.EVT_BUTTON, self.OnClick64)

        self.format = wx.StaticText(self, -1, "1024/50ms")  # current mode

        self.tbtn_video = wx.ToggleButton(self, id=-1, size=(100, -1), label="Stop Video")
        self.tbtn_video.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleVideo)

        btn_frame = wx.Button(self, id=-1, label="Snapshot")
        btn_frame.Bind(wx.EVT_BUTTON, self.onclickSnapshot)

        btn_cube = wx.Button(self, id=-1, label="Cube")
        btn_cube.Bind(wx.EVT_BUTTON, self.onclickCube)

        # New 2021-10-26
        self.cb_ring = wx.CheckBox(self, -1, "Process Cube?")
        self.ring_radius = wx.StaticText(self, -1, "n/a", size=(20, -1))
        self.ring_width = wx.StaticText(self, -1, "n/a", size=(20, -1))
        self.ring_xc = wx.StaticText(self, -1, "n/a", size=(20, -1))
        self.ring_yc = wx.StaticText(self, -1, "n/a", size=(20, -1))
        self.ring_coma = wx.StaticText(self, -1, "n/a", size=(20, -1))
        self.ring_angle = wx.StaticText(self, -1, "n/a", size=(20, -1))
        #

        # New 2021-11-03 to use with snapshot tool
        self.cb_obj = wx.CheckBox(self, -1, "Measure snapshot?")
        self.obj_distance_to_center_x = wx.StaticText(self, -1, "n/a", size=(30, -1))
        self.obj_distance_to_center_y = wx.StaticText(self, -1, "n/a", size=(30, -1))
        #

        btn_exit = wx.Button(self, id=-1, label="Exit")
        btn_exit.Bind(wx.EVT_BUTTON, onclickExit)

        self.sb = self.CreateStatusBar(4, style=wx.STB_DEFAULT_STYLE)
        self.sb.SetStatusWidths([300, 300, 300, -1])
        self.sb.SetStatusText('Video started on entry', 1)

        hbox_gain = wx.BoxSizer(wx.HORIZONTAL)
        hbox_gain.Add(st_gain, flag=wx.RIGHT, border=103)
        hbox_gain.Add(self.tctrl_gain, proportion=1)

        hbox_star = wx.BoxSizer(wx.HORIZONTAL)
        hbox_star.Add(st_star, flag=wx.RIGHT, border=103)
        hbox_star.Add(self.tctrl_star, proportion=1)

        hbox_numb = wx.BoxSizer(wx.HORIZONTAL)
        hbox_numb.Add(st_number, flag=wx.RIGHT, border=103)
        hbox_numb.Add(self.tctrl_number, proportion=1)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(self.tbtn_video, 0, wx.ALIGN_CENTER)
        hbox1.Add(btn_frame, 0, wx.ALIGN_CENTER)
        hbox1.Add(btn_cube)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(btn_1024, 0, wx.ALIGN_CENTER)
        hbox2.Add(btn_256, 0, wx.ALIGN_CENTER)
        hbox2.Add(btn_64, 0, wx.ALIGN_CENTER)

        hbox_format = wx.BoxSizer(wx.HORIZONTAL)
        hbox_format.Add(self.format, flag=wx.RIGHT, border=103)

        gs = wx.GridSizer(3, 5, 5, 5)
        gs.Add(wx.StaticText(self, -1, 'Ring:'), 0, wx.EXPAND)
        gs.Add(wx.StaticText(self, -1, 'Radius:'), 0, wx.EXPAND)
        gs.Add(self.ring_radius, 0, wx.EXPAND)
        gs.Add(wx.StaticText(self, -1, 'Width:'), 0, wx.EXPAND)
        gs.Add(self.ring_width, 0, wx.EXPAND)
        gs.Add(wx.StaticText(self, -1, ' '), 0, wx.EXPAND)
        gs.Add(wx.StaticText(self, -1, 'Center X:'), 0, wx.EXPAND)
        gs.Add(self.ring_xc, 0, wx.EXPAND)
        gs.Add(wx.StaticText(self, -1, 'Center Y:'), 0, wx.EXPAND)
        gs.Add(self.ring_yc, 0, wx.EXPAND)
        gs.Add(wx.StaticText(self, -1, ' '), 0, wx.EXPAND)
        gs.Add(wx.StaticText(self, -1, 'Coma:'), 0, wx.EXPAND)
        gs.Add(self.ring_coma, 0, wx.EXPAND)
        gs.Add(wx.StaticText(self, -1, 'Angle:'), 0, wx.EXPAND)
        gs.Add(self.ring_angle, 0, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(10)
        vbox.Add(hbox_gain, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)
        vbox.Add(hbox_star, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)
        vbox.Add(hbox_numb, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)
        vbox.AddSpacer(10)

        vbox.Add(hbox2)
        vbox.Add(hbox_format)
        vbox.AddSpacer(30)
        vbox.Add(hbox1)
        vbox.AddSpacer(20)

        # New
        vbox.Add(self.cb_ring)
        vbox.Add(gs)
        vbox.AddSpacer(20)
        #

        # New 2021-11-03
        obj_sizer = wx.BoxSizer(wx.HORIZONTAL)
        obj_sizer.Add(self.cb_obj, 0, wx.ALIGN_CENTER)
        obj_sizer.AddSpacer(10)
        obj_sizer.Add(wx.StaticText(self, -1, "Xd:"), 0, wx.ALIGN_CENTER)
        obj_sizer.Add(self.obj_distance_to_center_x, 0, wx.ALIGN_CENTER)
        obj_sizer.AddSpacer(20)
        obj_sizer.Add(wx.StaticText(self, -1, "Yd:"), 0, wx.ALIGN_CENTER)
        obj_sizer.Add(self.obj_distance_to_center_y, 0, wx.ALIGN_CENTER)
        vbox.Add(obj_sizer)
        vbox.AddSpacer(20)
        #

        vbox.Add(btn_exit, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, border=20)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.videopPanel, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, border=40)
        sizer.Add(vbox)

        self.SetSizer(sizer)
        # self.Fit()   # This line works under Mac OS but no with
        sizer.Fit(self)
        self.Show(True)
        self.Layout()  # This is new too

        # dirname = os.path.abspath('/opt/local/lib/')  # Edison's macbook
        # library_file = os.path.join(dirname, 'libASICamera2.dylib')  # under Mac OS
        # library_file = os.path.abspath('/opt/local/lib/libASICamera2.dylib')  # under Mac OS
        library_file = os.path.abspath('/usr/local/lib64/libASICamera2.so')  # under Linux (NUCs)
        #library_file = os.path.abspath('/usr/local/lib/libASICamera2.so')  # under Ubuntu (Andrei's laptop)
        self.camera = ASIcamera(library_file)
        self.camera.open()

        if not self.camera.cap:
            print("Camera not initialized")
            quit()

        self.sb.SetStatusText(self.camera.status)
        self.camera.set(self.exposure_time, self.gain, self.brightness, self.bandwidth, self.roi_x0, self.roi_y0,
                        self.roi_width, self.roi_height)
        self.camera.check("After Set():")

        whbi = self.camera.cap.get_roi_format()
        self.sz = whbi[0] * whbi[1] * 2
        self.shape = [whbi[1], whbi[0]]  # for later
        self.timeout = (self.camera.cap.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500
        # Now capture can be started
        self.camera.cap.start_video_capture()

        # timer def was moved down here just after the whole setup
        self.videotimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnUpdateVideo_2, self.videotimer)
        self.videotimer.Start(100)  # 100ms refresh time in video mode

    def OnUpdateVideo_2(self, event):
        image_data = self.camera.GetVideoFrame()
        if image_data is None:
            return
        else:
            self.FrameDisplay(image_data)  # display frame

    def OnUpdateVideo(self, event):
        # timestamp = datetime.utcnow()
        # ########Debug
        # aux = timestamp - self.oldtime
        # self.oldtime = timestamp
        # ticktime = aux.seconds + aux.microseconds / 1e6
        ##############

        # Create buffer array
        image_buffer = bytearray(self.sz)

        try:
            asi._get_video_data(0, self.timeout, buffer_=image_buffer)
        except (asi.ZWO_Error, asi.ZWO_IOError, asi.ZWO_CaptureError) as e:
            self.sb.SetStatusText('camera _get_video_data error {}'.format(e), 0)
            # self.videotimer.Stop()

        image_data = np.ndarray(buffer=image_buffer, dtype=np.uint16, shape=(self.shape[0], self.shape[1], 1)) // 16
        self.FrameDisplay(image_data)  # display frame

    def OnClick1024(self, event):
        self.roi_width, self.roi_height = 1024, 1024
        self.roi_x0 = int((1920 - self.roi_width) / 2)
        self.roi_y0 = int((1080 - self.roi_height) / 2)
        self.exposure_time = EXP_TIME_1024
        self.bandwidth = 80
        self.format.SetLabel(f'1024/{self.exposure_time}ms')

    def OnClick256(self, event):
        self.roi_width, self.roi_height = 256, 256
        self.roi_x0 = int((1920 - self.roi_width) / 2)
        self.roi_y0 = int((1080 - self.roi_height) / 2)
        self.exposure_time = EXP_TIME_256
        self.bandwidth = 80
        self.format.SetLabel(f'256/{self.exposure_time}ms')

    def OnClick64(self, event):
        self.roi_width, self.roi_height = 64, 64
        self.roi_x0 = int((1920 - self.roi_width) / 2)
        self.roi_y0 = int((1080 - self.roi_height) / 2)
        self.exposure_time = EXP_TIME_64
        self.bandwidth = 100
        self.format.SetLabel(f'64/{self.exposure_time}ms')

    def VideoOff(self):
        self.videotimer.Stop()
        self.camera.cap.stop_video_capture()
        # self.video = 0
        self.sb.SetStatusText('Video stopped', 1)
        # event.GetEventObject().SetLabel("Start Video")
        self.tbtn_video.SetLabel("Start Video")
        # self.tbtn_video.SetValue(False)
        self.sb.SetStatusText(" ", 2)
        self.sb.SetStatusText(" ", 3)

    def VideoOn(self):
        # Apply GUI settings to camera
        self.GetValues()
        self.camera.set(self.exposure_time, self.gain, self.brightness, self.bandwidth, self.roi_x0, self.roi_y0,
                        self.roi_width, self.roi_height)
        self.camera.check("Set camera for Video:")

        # After camera set, these parameter must be set to use with asi._get_video_data()
        whbi = self.camera.cap.get_roi_format()
        self.sz = whbi[0] * whbi[1] * 2
        self.shape = [whbi[1], whbi[0]]  # for later
        self.timeout = (self.camera.cap.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500
        # Now capture can be started
        self.camera.cap.start_video_capture()

        self.videotimer.Start(100)
        self.sb.SetStatusText('Video started', 1)
        # event.GetEventObject().SetLabel("Stop Video")
        self.tbtn_video.SetLabel("Stop Video")
        # self.tbtn_video.SetValue(True)

    def OnToggleVideo(self, event):
        state = event.GetEventObject().GetValue()
        # print("Video: {}",self.video)
        if state:
            self.VideoOff()
        else:
            self.VideoOn()

    def GetValues(self):
        try:
            self.gain = int(self.tctrl_gain.GetValue())
            # print("gain:{}".format(self.gain))
        except ValueError:
            print("Error setting Gain, using {}".format(self.gain))
        try:
            self.star = self.tctrl_star.GetValue()
            # print("Star: "+self.star)
        except ValueError:
            text = "Wrong value for Star"
        try:
            self.number = int(self.tctrl_number.GetValue())
            # print("Number-exp:{}".format(self.number))
        except ValueError:
            text = "Wrong value for # of exposures"

    def FrameDisplay(self, image_data):  # display frame
        imax = np.max(image_data)
        imin = np.min(image_data)
        self.sb.SetStatusText('Imax={} Imin={}'.format(imax, imin), 2)
        frame16 = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)  # frame16.dtype es uint16 y frame16.shape es 3-ch
        frame8 = cv2.convertScaleAbs(frame16, alpha=(255 / imax))

        new_width = self.framewidth
        new_height = self.frameheight
        frame = cv2.resize(frame8, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        img_buf = wx.ImageFromBuffer(frame.shape[1], frame.shape[0], frame)
        self.bmp = wx.Bitmap(img_buf)  # self.bmp = wx.BitmapFromImage(img_buf)  is  deprecated
        self.Refresh()

    def onclickSnapshot(self, e):
        if self.videotimer.IsRunning():
            self.sb.SetStatusText('Timer is running! Stop Video Timer first before take a snapshot', 3)
            # self.videotimer.Stop()
            self.VideoOff()

        # Apply GUI settings to camera
        self.GetValues()
        self.camera.set(self.exposure_time, self.gain, self.brightness, self.bandwidth, self.roi_x0, self.roi_y0,
                        self.roi_width, self.roi_height)
        self.camera.check("Set camera for SnapShot:")

        timestamp = datetime.utcnow()
        '''
        try:
            image_data = self.camera.cap.capture() // 16   # image_data.dtype is uint16
        except (asi.ZWO_Error, asi.ZWO_IOError, asi.ZWO_CaptureError) as e:
            self.sb.SetStatusText('camera capture() error: {}'.format(e), 0)
            return
        '''
        image_data = self.camera.snapshot()
        if image_data.any():
            self.FrameDisplay(image_data)  # display frame
            filename = self.savedir + timestamp.strftime('%Y-%m-%d-%H%M%S') + '_snapshot' + '.fits'
            self.SaveFits(image_data, timestamp, filename)  # it writes original data
        else:
            print('Error taking SnapShot(), no data saved')
            self.sb.SetStatusText('camera capture() error', 0)

        self.VideoOn()  # resume video

    def onclickCube(self, e):
        if self.roi_width > 256:
            print("No cube in full-field mode!")
            return
        if self.videotimer.IsRunning():
            self.sb.SetStatusText('Timer is running! Stop Video Timer first before take a data cube', 3)
            # self.videotimer.Stop()
            self.VideoOff()

        # Apply GUI settings to camera
        self.GetValues()
        print("Cube is acquired with 1ms")
        # self.camera.set(self.exposure_time, self.gain, self.brightness, self.bandwidth, self.roi_x0, self.roi_y0,
        #                 self.roi_width, self.roi_height)
        self.camera.set(EXP_TIME_CUBE, self.gain, self.brightness, self.bandwidth, self.roi_x0, self.roi_y0,
                        self.roi_width, self.roi_height)
        self.camera.check("Set camera for cube:")

        N = self.number
        # get buffer size
        whbi = self.camera.cap.get_roi_format()
        shape = [whbi[1], whbi[0]]  # for later

        '''
        N = self.number
        timestamp = [None] * N
        # Create buffer array
        images_buffer = [None] * N
        # set buffer size
        whbi = self.camera.cap.get_roi_format()
        sz = whbi[0] * whbi[1] * 2
        shape = [whbi[1], whbi[0]]  # for later
        for i in range(N):
            images_buffer[i] = bytearray(sz)

        # get timeout
        timeout = (self.camera.cap.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500

        # Video capture
        dropped = 1
        current_attempt = 0
        for attempt in range(self.max_attempts):
            current_attempt = attempt

            self.camera.cap.start_video_capture()

            for i in range(N):
                timestamp[i] = datetime.utcnow()

                try:
                    # this is the fastest way to get the buffer data that minimizes drop frames
                    asi._get_video_data(0, timeout, buffer_=images_buffer[i])
                except (KeyboardInterrupt, SystemExit):
                    raise
                except (asi.ZWO_Error, asi.ZWO_IOError, asi.ZWO_CaptureError) as e:
                    self.sb.SetStatusText('camera _get_video_data error {}'.format(e), 0)
                    break

                if self.camera.cap.get_dropped_frames() > 0:
                    aux = (i, N, self.camera.cap.get_dropped_frames())
                    print('cube failed at iteration %d of %d with %d dropped frames' % aux)
                    break

            dropped = self.camera.cap.get_dropped_frames()
            self.camera.cap.stop_video_capture()

            if dropped == 0:
                break

        if current_attempt == (self.max_attempts - 1):
            self.sb.SetStatusText('Video failed', 3)
            return

        '''
        images_buffer, timestamp, dropped, current_attempt = self.camera.cube(self.number, self.max_attempts)

        self.sb.SetStatusText('data cube ends, %d dropped frames of %d at attempt %d of %d' % (dropped, N,
                                                                                               current_attempt + 1,
                                                                                               self.max_attempts), 1)

        # Now I will process (?) and save data
        # get images from buffer with shape defined up
        img_list = [None] * N
        for i in range(N):
            img_list[i] = np.frombuffer(images_buffer[i], dtype=np.uint16).reshape(shape) // 16

        #        self.image_segmentation(img_list[N-1])

        # we skip the first value of the timestamps. No stamps if N<5
        fps = 0
        if N > 5:
            ts = np.zeros(N - 2)
            for i in range(N - 2):
                aux = timestamp[i + 2] - timestamp[i + 1]
                ts[i] = aux.seconds + aux.microseconds / 1e6

            fps = 1 / ts.mean()
            print('fps {:.1f}'.format(fps))

        # Cast the list into a numpy array
        img_array = np.array(img_list)

        filename = self.savedir + timestamp[N - 1].strftime('%Y-%m-%d-%H%M%S') + '_cube' + '.fits'

        self.SaveFits(img_array, timestamp[N - 1], filename, nframes=N, fps=fps)

        # New 2021-10-26
        if self.cb_ring.GetValue():
            # file = os.path.basename(filename)
            # indir = os.path.dirname(filename)
            # outdir = "/home/ebustos/PycharmProjects/processcube/output"
# 2022-11-20 add wait period
            time.sleep(0.1)
            data, imv = cube2.Moments(img_array)
            print(data)
            self.ring_radius.SetLabel(f'{data["image"]["impar"][3]:.2f}')
            self.ring_width.SetLabel(f'{data["image"]["impar"][4]:.2f}')
            self.ring_xc.SetLabel(f'{data["image"]["impar"][5]:.2f}')
            self.ring_yc.SetLabel(f'{data["image"]["impar"][6]:.2f}')
            self.ring_coma.SetLabel(f'{data["image"]["impar"][9]:.2f}')
            self.ring_angle.SetLabel(f'{data["image"]["impar"][10]:.2f}')
        else:
            print("No voy a procesar")
        # end New

        self.VideoOn()  # resume video

    def SaveFits(self, image, tstamp, filename, nframes=0, fps=0):
        """
        # set fits header
        hdr = fits.Header()
        hdr['EXPOSURE'] = (str(self.camera.cap.get_control_value(asi.ASI_EXPOSURE)[0]), 'Exposure time in usec')
        hdr['GAIN'] = (str(self.camera.cap.get_control_value(asi.ASI_GAIN)[0]), 'The ratio of output / input')
        hdr['BRIGHT'] = (str(self.camera.cap.get_control_value(asi.ASI_BRIGHTNESS)[0]), 'Brightness or Offset of image')
        hdr['BANDWDTH'] = (str(self.camera.cap.get_control_value(asi.ASI_BANDWIDTHOVERLOAD)[0]), 'Bandwidth Overload')
        hdr['FLIP'] = (str(self.camera.cap.get_control_value(asi.ASI_FLIP)[0]), 'Flip mode')
        hdr['HIGHSPD'] = (str(self.camera.cap.get_control_value(asi.ASI_HIGH_SPEED_MODE)[0]), 'High speed mode')
        hdr['START_X'] = (str(self.camera.cap.get_roi()[0]), 'ROI X origin in pixels')
        hdr['START_Y'] = (str(self.camera.cap.get_roi()[1]), 'ROI Y origin in pixels')
        hdr['WIDTH'] = (str(self.camera.cap.get_roi()[2]), 'ROI width in pixels')
        hdr['HEIGHT'] = (str(self.camera.cap.get_roi()[3]), 'ROI height in pixels')
        hdr['BIN'] = (str(self.camera.cap.get_bin()), 'Binning factor x,y')
        hdr['CAMERA'] = (asi.list_cameras()[0], 'Camera Model')
        hdr['CAMTEMP'] = (str(self.camera.cap.get_control_value(asi.ASI_TEMPERATURE)[0] / 10),
                          'Camera Temperature in Celsius deg')
        hdr['COMMENT'] = (str(self.camera.cap.get_camera_property()))
        hdr['SITE'] = (self.Site, 'Site Location')
        hdr['DATE-OBS'] = (str(tstamp.strftime('%Y.%m.%dT%H:%M:%S.%f')), 'UTC start date of observation')
        hdr['SWCREATE'] = (self.APP_TITLE, 'Name of software')
        hdr['NFRAMES'] = (str(nframes), 'Data cube total frames')
        hdr['FPS'] = ('{:.1f}'.format(fps), 'Frames per second')
        hdr['STAR'] = self.star
        """

        self.camera.Save_Fits(image, tstamp, filename, self.Site, self.APP_TITLE, self.star, nframes, fps)

        txt = filename + ' saved'
        self.sb.SetStatusText(txt, 1)


if __name__ == '__main__':
    app = wx.App(0)
    myframe = MyFrame()
    app.MainLoop()
