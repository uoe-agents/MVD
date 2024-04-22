import imageio

class VideoRecorder(object):
    def __init__(self, height=480, width=480, fps=10):
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = enabled

    def record(self, env, camera=None):
        if self.enabled:
            frame = env.render(mode='rgb_array',
                               height=self.height,
                               width=self.width,
                               camera=camera)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            imageio.mimsave(file_name, self.frames, fps=self.fps)

    def reset(self):
        self.frames = []
