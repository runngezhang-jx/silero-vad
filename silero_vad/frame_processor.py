# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class FrameProcessor:
    def __init__(self, frame_size, padding=True):
        self.frame_size = frame_size
        # padding zeros for the last frame
        self.padding = padding
        self.remained_samples = np.array([])

    def add_frame(self, frame, is_last=False):
        self.remained_samples = np.concatenate((self.remained_samples, frame))
        while len(self.remained_samples) >= self.frame_size:
            frame = self.remained_samples[: self.frame_size]
            self.remained_samples = self.remained_samples[self.frame_size :]
            yield frame
        if is_last and self.padding:
            frame = self.remained_samples
            yield np.pad(frame, (0, self.frame_size - len(frame)))


if __name__ == "__main__":
    processor = FrameProcessor(frame_size=3)
    frames = [[1, 2, 3], [4, 5], [6, 7, 8]]
    for idx, frame in enumerate(frames):
        for frame in processor.add_frame(frame, idx == len(frames) - 1):
            print(frame)
