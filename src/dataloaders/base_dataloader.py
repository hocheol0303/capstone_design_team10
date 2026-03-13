import tensorflow as tf

class Augmentation:
    '''
    당장은 (원본 + landmark) input 받는건 염두에 두기만 하고 구현은 나중에

    이미지는 0~1 사이 값으로 들어와서 0~1 사이 값으로(clip함수) 출력한다고 가정
        한 번에 여러개 쓰는걸 고려해서 이렇게 하는데 만약 하나씩만 쓴다면 255 곱하는걸 넣어야지
    @staticmethod: 객체 없이 호출 가능
    Augmentation() 안만들어도 Augmentation.func() 가능
    tf.cond(): p의 확률로 앞에꺼, 1-p의 확률로 뒤에꺼
    밝기 조절하는거니까 hsv로 변환하고 v를 건드리자
    '''
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
        ]
    @staticmethod
    def apply_aug(img, aug_list):
        # 여기 있는건 0~1 사이 값임 -> 감안해서 변환할 것
        rgb_dispatch = {
            'gaussian_noise': Augmentation.gaussian_noise,
            'poisson_noise': Augmentation.poisson_noise,
            'flip': Augmentation.flip
        }
        hsv_dispatch = {
            'hsv_random_brightness': Augmentation.hsv_random_brightness,
            'hsv_random_contrast': Augmentation.hsv_random_contrast,
            'hsv_gamma_correction': Augmentation.hsv_gamma_correction
        }
        for aug_name in aug_list:
            if aug_name in rgb_dispatch:
                img = rgb_dispatch[aug_name](img)
        
        use_hsv = ('hsv_random_brightness' in aug_list or
                   'hsv_random_contrast' in aug_list or
                   'hsv_gamma_correction' in aug_list)
        if use_hsv:
            # hsv <-> rgb 변환을 한 번씩만 하기 위해 여기서 수행
            hsv = tf.image.rgb_to_hsv(img)
            h, s, v = tf.split(hsv, 3, axis=-1)
            for aug_name in aug_list:
                if aug_name in hsv_dispatch:
                    v = hsv_dispatch[aug_name](v)
            v = tf.clip_by_value(v, 0.0, 1.0)
            new_hsv = tf.concat([h, s, v], axis=-1)
            img = tf.image.hsv_to_rgb(new_hsv)
        # hsv - rgb 변환 시에도 값이 튈 수 있으므로 클리핑
        return tf.clip_by_value(img, 0.0, 1.0)
    @staticmethod
    def flip(img, p=0.5):
        return tf.cond(tf.random.uniform([]) < p,
                      lambda: tf.image.flip_left_right(img),
                      lambda: img
        )
    @staticmethod
    def rgb_random_brightness(img, p=0.5, max_delta=0.2):
        '''
        max_delta: 밝기 조절 범위(위아래로), 0~1 사이 값(더해지는 값)
        '''
        img = tf.cond(tf.random.uniform([]) < p,
                        lambda: tf.image.random_brightness(img, max_delta=max_delta),
                        lambda: img
        )
        return tf.clip_by_value(img, 0.0, 1.0)
    @staticmethod
    def hsv_random_brightness(v, p=0.3, max_delta=0.2):
        def adjust(delta):
            adjusted_v = tf.clip_by_value(v + delta, 0.0, 1.0)
            return adjusted_v
        return tf.cond(tf.random.uniform([]) < p,
                      lambda: adjust(tf.random.uniform([], -max_delta, max_delta)),
                      lambda: v
        )
    @staticmethod
    def rgb_random_contrast(img, p=0.5, lower=0.7, upper=1.3):
        '''
        값의 범위를 지정하여 대비 조절
        image = (image - mean) * contrast_factor + mean,   contrast_factor = [lower, upper] 범위 내 임의의 값
        '''
        img = tf.cond(tf.random.uniform([]) < p,
                        lambda: tf.image.random_contrast(img, lower=lower, upper=upper),
                        lambda: img
        )
        return tf.clip_by_value(img, 0.0, 1.0)
    @staticmethod
    def hsv_random_contrast(v, p=0.3, lower=0.7, upper=1.3):
        '''
        밝기 얘기를 계속 듣다보니 rgb보단 hsv의 v만 건드는게 좋지 않을까?
        hsv 색공간에서 변환하는게 색상 정보는 더 잘 보존한대(Gemini)
        '''
        def adjust(contrast_factor):
            v_mean = tf.reduce_mean(v)
            adjusted_v = (v-v_mean) * contrast_factor + v_mean
            adjusted_v = tf.clip_by_value(adjusted_v, 0.0, 1.0)
            return adjusted_v
        return tf.cond(tf.random.uniform([]) < p,
                      lambda: adjust(tf.random.uniform([], lower, upper)),
                      lambda: v
        )
    @staticmethod
    def gamma_correction(img, p=0.5, gamma=1.2):
        '''
        image의 pixel값은 0~1 사이로 들어옴
        image = image ** gamma * gain
        gamma > 1: 어두운 부분의 대비가 강조되고 전체적으로 밝아짐
        gamma < 1: 밝은 부분의 대비가 강조되고 전체적으로 어두워짐
            0.5**2 = 0.25
            0.5**0.5 = 0.707
            gamma 값이 작을수록 픽셀 값이 커짐 -> 밝아짐
            비선형적으로 변환됨

            0.2**2 = 0.04 (20% 밝기)
            0.3**2 = 0.09 (30% 밝기)
            0.4**2 = 0.16 (40% 밝기)
            0.5**2 = 0.25 (50% 밝기)
            0.6**2 = 0.36 (60% 밝기)
            0.7**2 = 0.49 (70% 밝기)
            0.8**2 = 0.64 (80% 밝기)
            어두운 영역의 대비가 낮아짐(뭉게짐)

            0.2**0.5 = 0.447
            0.3**0.5 = 0.548
            0.4**0.5 = 0.632
            0.5**0.5 = 0.707
            0.6**0.5 = 0.775
            0.7**0.5 = 0.837
            0.8**0.5 = 0.894
            어두운 영역의 대비가 높아짐(강조됨)

        gain: 이미지의 전체적인 밝기 조절 역할 = brightness가 맡기로 하자
        '''
        img = tf.cond(tf.random.uniform([]) < p,
                      lambda: tf.image.adjust_gamma(img, gamma=gamma, gain=1.0),
                      lambda: img
        )
        return tf.clip_by_value(img, 0.0, 1.0)
    @staticmethod
    def hsv_gamma_correction(v, p=0.2, gamma_lower=0.8, gamma_upper=1.2):
        '''
        hsv 색공간에서 v만 건드리는 감마 보정 -> contrast 하다보니 Gemini가 이것도 HSV를 추천함
        '''
        def adjust(gamma):
            adjusted_v = tf.math.pow(v, gamma)
            adjusted_v = tf.clip_by_value(adjusted_v, 0.0, 1.0)
            return adjusted_v
        return tf.cond(tf.random.uniform([]) < p,
                        lambda: adjust(tf.random.uniform([], gamma_lower, gamma_upper)),
                        lambda: v
        )
    @staticmethod
    def random_gamma_correction(img, p=0.5, gamma_lower=0.8, gamma_upper=1.2):
        def adjust():
            img = tf.image.adjust_gamma(img, gamma=tf.random.uniform([], gamma_lower, gamma_upper), gain=1.0)
            return tf.clip_by_value(img, 0.0, 1.0)
        return tf.cond(tf.random.uniform([]) < p, adjust, lambda: img)
    @staticmethod
    def hist_equalization(img_hsv_float, p=0.5):
        '''
        단순히 input 이미지 전체에 HE 적용
        배경 영역까지 다 입력하면 이상해질 수 있으므로 얼굴 영역에만 적용하는게 좋을 것 같아보임
        '''
        def _hist_eq():
            h, s, v = tf.split(img_hsv_float, num_or_size_splits=3, axis=-1)
            int_v = tf.cast(v*255, tf.int32)
            bins = tf.math.bincount(
                int_v,
                minlength=256,
                maxlength=256,
                dtype=tf.float32
            )
            pdf = bins / tf.reduce_sum(bins)
            cdf = tf.cumsum(pdf)

            equalized_v = tf.gather(cdf, int_v)
            equalized_v = tf.reshape(equalized_v, h.shape)
            equalized_v = tf.cast(equalized_v, h.dtype)
            equalized_img = tf.concat([h, s, equalized_v], axis=-1)
            equalized_img = tf.image.hsv_to_rgb(equalized_img)
            return equalized_img
        return tf.cond(tf.random.uniform([]) < p, lambda: _hist_eq(), lambda: img_hsv_float)
    @staticmethod
    def hist_equalization_with_landmark(img_hsv_float, p=0.5, landmarks=None):
        '''
        mediapipe에서 landmark를 미리 뽑아서 얼굴 영역만 HE 적용
        '''
        def _hist_eq():
            height, width, _ = img_hsv_float.shape
            face_coords = tf.constant([
                [int(landmarks[i].x*width), int(landmarks[i].y*height)] for i in Augmentation.FACE_OVAL
            ], tf.int32)
            x_min = tf.reduce_min(face_coords[:, 0])
            x_max = tf.reduce_max(face_coords[:, 0])
            y_min = tf.reduce_min(face_coords[:, 1])
            y_max = tf.reduce_max(face_coords[:, 1])
            
            face_box_hsv = img_hsv_float[y_min:y_max, x_min:x_max, :]
            equalized_img = Augmentation.hist_equalization(face_box_hsv, p=1.0)
            return equalized_img
        return tf.cond(tf.random.uniform([]) < p, lambda: _hist_eq(), lambda: img_hsv_float)
    @staticmethod
    def clahe():
        '''
        HE에 비해 코드 너무 복잡해짐 -> 보류
        '''
        pass
    @staticmethod
    def gaussian_noise(img, p=0.2, stddev=0.03):
        # rgb에서 조작하쟈
        def _add_noise():
            noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=stddev)
            return tf.clip_by_value(img + noise, 0.0, 1.0)
        return tf.cond(tf.random.uniform([]) < p, _add_noise, lambda: img)
    @staticmethod
    def poisson_noise(img, p=0.5, peak=100.0):
        def _add_noise():
            img_scaled = img * peak
            noise_img = tf.random.poisson(shape=[], lam=img_scaled)
            return tf.clip_by_value(noise_img / peak, 0.0, 1.0)
        return tf.cond(tf.random.uniform([]) < p, _add_noise, lambda: img)