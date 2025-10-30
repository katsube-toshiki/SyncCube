import datetime
import cv2
import numpy as np
import os
import glob

# 顔認識ライブラリをインポート
from insightface.app import FaceAnalysis


# 顔認識モデルのクラス
class Model:
    # モデルの初期化。顔認識ライブラリ「Insight Face」のオブジェクトを引数にとる
    def __init__(self, analyzer):
        self.analyzer = analyzer

        # 作成したモデルの保存場所
        self.binary_path = 'twice_bin'

        # モデル
        self.known_face_encodings = []
        self.known_face_names = []

        print("initing the models...")

        if os.path.exists(os.path.join(self.binary_path, 'encodings.npy')):
            # もしすでにモデルを作成済みであればそれを読み込む
            self.load_model()
        else:
            # なければ作る
            self.create_model()

    # 作成済みのモデルを読み込む
    def load_model(self):
        self.known_face_encodings = np.load(os.path.join(self.binary_path, 'encodings.npy'))
        self.known_face_names = np.load(os.path.join(self.binary_path, 'names.npy'))
    
    # 顔認識モデルを作成する
    def create_model(self):
        # 認識したい人の写真が入ったフォルダ
        people_image_dir_path = 'twice'

        # 写真のパスを取得する
        paths = map(os.path.abspath, glob.glob(people_image_dir_path + '/**', recursive=False))
        paths = filter(os.path.isdir, paths)

        # それぞれの写真について、analyzerでエンコードしてリストに入れていく
        for path in paths:
            files = os.listdir(path)
            print(files)
            
            for filename in files:
                known_people_image = cv2.imread(os.path.join(path, filename))
                known_people_faces = self.analyzer.get(known_people_image)

                if len(known_people_faces) < 1:
                    print(filename)
                    continue

                known_people_image_encoding = known_people_faces[0].embedding
                self.known_face_encodings.append(known_people_image_encoding)
                self.known_face_names.append(path.split('/')[-1])
            print(path.split('/')[-1] + ' done')
        
        # 使い回せるようにファイルとしてモデルを保存する
        np.save(os.path.join(self.binary_path, 'encodings.npy'), np.array(self.known_face_encodings))
        np.save(os.path.join(self.binary_path, 'names.npy'), np.array(self.known_face_names))
        
        print('binary files were saved.')
    
    # 顔の似ている度合いを計算する
    def compute_sim(self, encoding1, encoding2):
        return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))


# 動画を管理するクラス
class Movie:
    '''
    video_path: 動画ファイルのパス
    start_frame_no: スタートのフレーム番号
    step_frames: 一回の読み込みで進むフレーム数
    end_frame_no: 終わりのフレーム番号
    '''
    def __init__(self, video_path, start_frame_no=0, step_frames=1, end_frame_no=1e9):

        # OpenCVで動画を読み込む
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            raise Exception
        print('successfully video was loaded.')

        # 動画のプロフパティを取得する
        self.total_frames = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.end_frame_no = min(self.total_frames, end_frame_no)
        self.frame_rate = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        
        if start_frame_no < 0 or start_frame_no > self.end_frame_no:
            raise Exception
        
        # 現在のフレームを、start_frame_noに設定する
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

        self.step_frames = step_frames
        self.current_frame_no = start_frame_no

    # 現在のフレームを取得する
    def get_frame(self):
        # step_framesだけ進んだところに現在のフレームを設定する
        if self.step_frames != 1:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_no)

        ret, frame = self.video_cap.read()
        if not ret:
            raise Exception
        
        return Frame(frame, self.current_frame_no)
    
    # 次のフレームに移る
    def next(self):
        self.current_frame_no += self.step_frames

        # 最後のフレームならFalseを返す
        return self.current_frame_no < self.end_frame_no
    
    # リソースを解放する
    def release(self):
        self.video_cap.release()


# フレームを管理するクラス
class Frame:

    '''
    frame: openCVで用いられるframeオブジェクト(numpy配列)
    frame_no: 動画内での、そのフレームの番号
    '''
    def __init__(self, frame, frame_no):
        self.frame = np.array(frame)

        # openCVではBGRなので、RGBに直す
        self.rgb_frame = np.array(frame[:,:,::-1])

        # プロパティを取得する
        self.height, self.width, _ = self.frame.shape
        aspect_gcd = np.gcd(self.height, self.width)
        self.aspect_height = int(self.height / aspect_gcd)
        self.aspect_width = int(self.width / aspect_gcd)
        self.frame_no = frame_no
    
    '''フレーム内にある顔を検出する
    model: Modelクラスのインスタンス
    '''
    def face_recognize(self, model):
        insight_faces = model.analyzer.get(self.rgb_frame)
        self.face_locations = [insight_face.bbox.astype(np.int32) for insight_face in insight_faces]
        self.face_encodings = [insight_face.embedding for insight_face in insight_faces]

        self.faces = [Face(location, encoding, model) for location, encoding in zip(self.face_locations, self.face_encodings)]
        self.face_centers = np.array([face.center for face in self.faces])

        # 検出できたらTrueを返す
        return len(self.face_locations) > 0
    
    # 基準位置を中心にmagだけ拡大したframeを返す
    def zoom(self, center_x, center_y, mag):
        
        if mag < 1:
            return self
        
        mag2 = int(np.round(self.width*mag/self.aspect_width))
        new_width = mag2*self.aspect_width
        new_height =mag2*self.aspect_height

        new_center_x = int(center_x * mag)
        new_center_y = int(center_y * mag)

        size_after = (new_width, new_height)
        resized_frame = cv2.resize(self.frame, dsize=size_after)

        new_top = int(new_center_y + self.height/2)
        new_bottom = new_top - self.height
        new_right = int(new_center_x + self.width/2)
        new_left = new_right - self.width

        if new_top > new_height:
            new_top = new_height
            new_bottom = new_top - self.height
        if new_bottom < 0:
            new_bottom = 0
            new_top = new_bottom + self.height
        if new_right > new_width:
            new_right = new_width
            new_left = new_right - self.width
        if new_left < 0:
            new_left = 0
            new_right = new_left + self.width

        final_frame = resized_frame[new_bottom:new_top, new_left:new_right]
        return Frame(final_frame, self.frame_no)
    
    # imshowでframeを表示する(検証用)
    def display(self, window_name='frame'):
        cv2.imshow(window_name, cv2.resize(self.frame, dsize=(960, 540)))

        # キー入力を返す
        key = cv2.waitKey(0)
        return key != ord('q')


# 顔を管理するクラス
class Face:
    '''
    location: frame内での位置。(left, top, right, bottom)
    encoding: 顔情報
    model: Modelクラスのインスタンス
    '''
    def __init__(self, location, encoding, model):
        self.location = location
        self.encoding = encoding
        
        left, top, right, bottom = self.location
        self.center = np.array(((top + bottom)/2, (left + right)/2))

        # それぞれの人に対する、顔の類似度
        self.similarity_onebyone = {name: 0 for name in model.known_face_names}

        self.similarities = [model.compute_sim(known_face_encoding, self.encoding) for known_face_encoding in model.known_face_encodings]
        for name, similarity in zip(model.known_face_names, self.similarities):
            self.similarity_onebyone[name] += similarity
        
        # 一番類似度が高かった人の名前、類似度
        self.bestmatch_name, self.bestmatch_similarity = max(self.similarity_onebyone.items(), key=lambda x: x[1])


# 顔の、フレーム番号と位置を管理する
class TimeLocation:
    def __init__(self, frame_no, location):
        self.frame_no = frame_no
        self.location = location


'''
顔のシーケンス(連続)を管理するクラス

シーケンスとは：
フレーム間において、位置が近い顔は同一のものである可能性が高い。そのような位置が近い顔をシーケンスという形で管理し、名前の特定はシーケンスごとに行う
'''
class FaceSequence:
    def __init__(self, face, frame_no):
        self.faces = [face]
        self.time_locations = [TimeLocation(frame_no, face.location)]

        # 現在の顔の位置の、中心位置
        self.current_center = face.center

        # 顔の移動速度
        self.velocity = None

        # 次の顔の予想位置
        self.estimated_next_center = face.center

        # 次の顔が見つからなかった回数
        self.count_not_found = 0
        self.name = None

        # 検出された顔の名前を管理する変数
        self.max_name_count = 0
        self.counter = None

    # シーケンスに顔を追加する
    def update(self, face, frame_no):

        self.faces.append(face)
        self.time_locations.append(TimeLocation(frame_no, face.location))

        # 次の顔の位置を予想する
        previous_center = self.current_center
        self.current_center = face.center
        self.velocity = self.current_center - previous_center
        self.estimated_next_center = self.current_center + self.velocity

        # 顔が見つかったことを記録する
        self.count_not_found = 0
    
    # 毎フレームで実行し、顔が見つからなかった場合はシーケンスを終了する
    def clock(self):
        if self.count_not_found > 0:
            return False
        self.count_not_found += 1
        return True
    
    # faceの位置と、顔の予想位置との距離を計算する
    def distance(self, face):
        return np.linalg.norm(self.estimated_next_center - face.center)
    
    # 他のシーケンスと繋がりがあれば、シーケンス同士を繋げる
    def connect(self, face_sequence):
        start_time_location = self.time_locations[-1]
        end_time_location = face_sequence.time_locations[0]

        print("connect {} -> {} ({})".format(start_time_location.frame_no, end_time_location.frame_no, self.name))

        gap_frames = end_time_location.frame_no - start_time_location.frame_no - 1
        for i in range(1, gap_frames+1):
            location = [x1 + (x2 - x1) / gap_frames * i for x1, x2 in zip(start_time_location.location, end_time_location.location)]
            frame_no = start_time_location.frame_no + i
            time_location = TimeLocation(frame_no, location)
            self.time_locations.append(time_location)

        self.time_locations.extend(face_sequence.time_locations)

    # シーケンス全体で、その人の名前を特定する
    def identify_name(self):
        self.counter = {}

        # シーケンスに含まれている顔全てで、名前を集計する
        for face in self.faces:
            if face.bestmatch_name in self.counter.keys():
                self.counter[face.bestmatch_name] += 1
            else:
                self.counter[face.bestmatch_name] = 1
        
        print(self.counter)
        
        # 最も多かった名前を採用する
        name, count = max(self.counter.items(), key=lambda x: x[1])

        self.name = name
        self.max_name_count = count


'''
シーンを管理するクラス
シーンとは：
フレームのまとまりである。連続的なつながりのあるフレームは同じシーンに属する。
動画に映っている場面が変わった場合にシーンも新しいものにする。
'''
class Scene:
    # 初期化処理
    def __init__(self, scene_name, first_frame_no, model):
        self.first_frame_no = first_frame_no
        self.frames = []
        self.member_frames = {}
        self.scene_name = scene_name
        self.face_sequences = []
        self.active_face_sequences = []
        self.model = model

        for name in model.known_face_names:
            self.member_frames[name] = []

        print(self.scene_name, 'was created')
    
    # frameがそのシーンに含まれるかどうか判断する
    def is_same(self, frame):

        # シーン最初のフレームならTrueを返す
        if self.first_frame_no == frame.frame_no:
            return True
        
        # 顔が一つもなければFalseを返す
        if len(self.active_face_sequences) == 0 or len(frame.faces) == 0:
            return False
        
        # 100フレームを超えたらFalseを返す
        if len(self.frames) > 100:
            return False
        
        # frameの顔のうち、最も近くにあるシーンの顔との距離が一定以上だと、違うシーンだと判断する
        distances = []
        for face in frame.faces:
            distances.append(min([face_sequence.distance(face) for face_sequence in self.active_face_sequences]))
        min_distance = min(distances)

        return min_distance < 30
    

    # シーンにframeを追加する
    def add(self, frame):
        self.frames.append(frame)
        for frames in self.member_frames.values():
            frames.append(frame)

        # シーンの最初のフレームならシーケンスにどんどん入れていく
        if len(self.frames) == 1:
            self.active_face_sequences = [FaceSequence(face, frame.frame_no) for face in frame.faces]
        else:

            # FaseSequenceにframe内のfaceを追加していく
            distances_list = [0]*len(frame.faces)
            for i, face in enumerate(frame.faces):
                distances = [face_sequence.distance(face) for face_sequence in self.active_face_sequences]
                distances_list[i] = distances

            for i, face in enumerate(frame.faces):
                distances = distances_list[i]
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                if min_distance < 100:
                    self.active_face_sequences[min_idx].update(face, frame.frame_no)
                else:
                    self.active_face_sequences.append(FaceSequence(face, frame.frame_no))

            new_face_sequences = []
            for face_sequence in self.active_face_sequences:
                if not face_sequence.clock():
                    self.face_sequences.append(face_sequence)
                else:
                    new_face_sequences.append(face_sequence)
            self.active_face_sequences = new_face_sequences
    
    # シーンに含まれるFaceSequenceの、名前を特定する
    def identify_names_and_edit_frame(self):
        used = {}

        for face_sequence in self.active_face_sequences:
            self.face_sequences.append(face_sequence)
        
        for face_sequence in self.face_sequences:
            face_sequence.identify_name()

        self.face_sequences.sort(key=lambda x: -len(x.time_locations))

        # FaceSequenceの、connectを行なっていく
        for i, face_sequence1 in enumerate(self.face_sequences):
            for j, face_sequence2 in enumerate(self.face_sequences):
                if i == j:
                    continue
                if face_sequence1.name != face_sequence2.name:
                    continue
                time_locations1 = face_sequence1.time_locations
                time_locations2 = face_sequence2.time_locations
                if time_locations1[-1].frame_no >= time_locations2[0].frame_no:
                    continue
                if time_locations2[0].frame_no - time_locations1[-1].frame_no > 10:
                    continue
                face_sequence1.connect(face_sequence2)

        self.face_sequences.sort(key=lambda x: -len(x.time_locations))

        for face_sequence in self.face_sequences:

            # 短すぎるFaceSequenceは消去する
            if face_sequence.max_name_count < 3:
                print("discard {}".format(face_sequence.counter))
                continue

            name = face_sequence.name
            for time_location in face_sequence.time_locations:
                left, top, right, bottom = time_location.location
                frame_no = time_location.frame_no

                if (name, frame_no) in used.keys():
                    continue

                frame = self.member_frames[name][frame_no - self.first_frame_no]

                # 顔をズーム
                mag = frame.height / (bottom-top) / 3
                zoomed_frame = frame.zoom((right+left)/2, (top+bottom)/2, mag)

                self.member_frames[name][frame_no - self.first_frame_no] = zoomed_frame

                used[(name, frame_no)] = True

    # imshowでシーンを表示する（検証用）
    def display(self):
        for frame in self.frames:
            if not frame.display():
                return False
        return True
    
    # シーンを動画に書き出す
    def write_video(self, video_writers):
        for video_writer in video_writers:
            frames = self.member_frames[video_writer.name]
            for frame in frames:
                video_writer.write(frame)


# openCVを用いてビデオライターを管理するクラス
class VideoWriter:
    def __init__(self, dir, name, frame_rate, width, height):
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.name = name
        self.writer = cv2.VideoWriter('{}/{}.mp4'.format(dir, name), fmt, frame_rate, (width, height))
    
    def write(self, frame):
        self.writer.write(frame.frame)
    
    def release(self):
        self.writer.release()



if __name__ == '__main__':

    # ライブラリ「Insight Face」のインスタンスを作成する
    analyzer = FaceAnalysis()
    analyzer.prepare(ctx_id=0)

    # Insight Faceを用いて、顔認識モデルを作成する
    model = Model(analyzer)

    # 動画を読み込む
    movie = Movie('heartshaker.mp4', 1000, 1, 1200)

    # 動画を保存するフォルダを作成する
    now = datetime.datetime.now()
    dir_name = now.strftime('%Y%m%d%H%M%S')
    os.mkdir(dir_name)

    # 認識する人それぞれに対応するビデオライターを作成する
    video_writers = [VideoWriter(dir_name, name, movie.frame_rate, movie.width, movie.height) for name in model.known_face_names]

    # 最初のシーンを作成する
    scene_no = 0
    scene = Scene('scene {}'.format(scene_no), movie.current_frame_no+1, model)

    try:
        # 動画が終わるまでループを回す
        while movie.next():
            
            # 今のフレームの番号を出力する
            print(movie.current_frame_no, '/', movie.total_frames)

            # 今のフレームを動画から取得する
            frame = movie.get_frame()
            frame_no = frame.frame_no

            # フレームにある顔を認識する
            frame.face_recognize(model)
            
            if scene.is_same(frame):
                # フレームが現在のシーンに含まれる場合、フレームをシーンに追加する
                scene.add(frame) 
            else:
                # シーンに含まれる顔の名前を特定し、顔をズームする
                scene.identify_names_and_edit_frame()

                # シーンをビデオライターで動画に書き出す
                scene.write_video(video_writers)

                # シーンを新しく作る
                scene_no += 1
                scene = Scene('scene {}'.format(scene_no), movie.current_frame_no, model)
                scene.add(frame)

        # 最後のシーンを処理する
        scene.identify_names_and_edit_frame()
        scene.write_video(video_writers)

    except Exception as e:
        print(e)

    finally:
        # 動画オブジェクト、ビデオライターオブジェクトを解放する
        movie.release()
        for writer in video_writers:
            writer.release()