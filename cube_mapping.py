import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
import sys
import time
import pygame
import serial

# 開始時間を記録しておく用
start_time = None

# fpsを記録しておく用
fps = None

# 読み込んだ動画を入れるリスト
videos = []

# 現在のフレームを記録しておく用
current_frame = 0

# 視点の原点からの距離
distance = 5.0

# ピッチ、ヨー、ロール
r = [0]*3

# メンバーの名前
names_dict = {
    "heartshaker": ["chaeyoung", "jihyo", "nayeon", "tzuyu", "dahyun", "momo"],
    "dynamite": ["jhope", "jungkook", "jin", "v", "jimin", "rm"],
}
names = None

# マイコン接続用
ser = None

vertex = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
]
texture = [
    [1.0, 1.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [1.0, 0.0],
]
textures = [0] * 6
face = [
    [0, 1, 2, 3],
    [1, 5, 6, 2],
    [5, 4, 7, 6],
    [4, 0, 3, 7],
    [4, 5, 1, 0],
    [7, 3, 2, 6],
]

# 最初に行う処理
def init(title):
    global textures
    global ser
    global names_dict
    global names
    global videos
    global fps

    # OpenGLの処理
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glEnable(GL_TEXTURE_2D)
    textures = glGenTextures(6)
    glDisable(GL_TEXTURE_2D)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_FRONT)

    names = names_dict[title]

    # 動画を読み込む
    for i in range(6):
        filepath = "{}/".format(title) + names[i] + ".mp4"
        videos.append(cv2.VideoCapture(filepath))
    fps = videos[0].get(cv2.CAP_PROP_FPS)
    print('setup src_img')

    # マイコンと接続する
    ser = serial.Serial('/dev/ttyUSB0', 9600)
    ser.reset_input_buffer()

    # 音声を再生し始める
    pygame.mixer.init()
    pygame.mixer.music.load("{}.mp3".format(title))
    pygame.mixer.music.play(loops=-1)

# 定期的に実行する関数
def idle():
    global textures
    global r
    global current_frame
    global start_time
    global fps
    global names

    print(r)
    
    # 角度を、0~360度の間になるように調整する
    a = round(r[1])
    b = round(r[0])
    while a < 0:
        a += 360
    while a >= 360:
        a -= 360
    while b < 0:
        b += 360
    while b >= 360:
        b -= 360
    
    # 立方体の面に動画のフレームを貼っていく
    glEnable(GL_TEXTURE_2D)
    for i in range(6):
        # フレームを読み込む
        ret, frame = videos[i].read()

        # 最後のフレームだった場合、最初のフレームに戻る
        if not ret:
            videos[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = videos[i].read()
            current_frame = 0
            start_time = time.perf_counter()

        # 視点からは見えない面は、処理の軽量化のために飛ばす
        if a == 0 or a == 180:
            if i == 1 or i == 3:
                continue
        if a == 90 or a == 270:
            if i == 0 or i == 2:
                continue
        if b == 0 or b == 180:
            if i == 4 or i == 5:
                continue
        if b == 90:
            if i != 5:
                continue
        if b == 270:
            if i != 4:
                continue
        if (0 <= a <= 90 and 0 <= b <= 90) or (180 <= a <= 270 and 90 <= b <= 180):
            if not (i == 2 or i == 3 or i == 5):
                continue
        if (0 <= a <= 90 and 90 <= b <= 180) or (180 <= a <= 270 and 0 <= b <= 90):
            if not (i == 0 or i == 1 or i == 5):
                continue
        if (0 <= a <= 90 and 180 <= b <= 270) or (180 <= a <= 270 and 270 <= b <= 360):
            if not (i == 0 or i == 1 or i == 4):
                continue
        if (0 <= a <= 90 and 270 <= b <= 360) or (180 <= a <= 270 and 180 <= b <= 270):
            if not (i == 2 or i == 3 or i == 4):
                continue
        if (90 <= a <= 180 and 0 <= b <= 90) or (270 <= a <= 360 and 90 <= b <= 180):
            if not (i == 0 or i == 3 or i == 5):
                continue
        if (90 <= a <= 180 and 90 <= b <= 180) or (270 <= a <= 360 and 0 <= b <= 90):
            if not (i == 1 or i == 2 or i == 5):
                continue
        if (90 <= a <= 180 and 180 <= b <= 270) or (270 <= a <= 360 and 270 <= b <= 360):
            if not (i == 1 or i == 2 or i == 4):
                continue
        if (90 <= a <= 180 and 270 <= b <= 360) or (270 <= a <= 360 and 180 <= b <= 270):
            if not (i == 0 or i == 3 or i == 4):
                continue

        # フレームを立方体の面に貼る
        name = names[i]
        img = convert_size(frame, (640, 640), name)
        h, w = img.shape[:2]
        glBindTexture(GL_TEXTURE_2D, textures[i])
        gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, w, h, GL_RGB, GL_UNSIGNED_BYTE, img)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glDisable(GL_TEXTURE_2D)

    # OpenGLに再描画を要求する
    glutPostRedisplay()

    # 動画の遅れ、または進みを補正する
    elapsed_time = time.perf_counter() - start_time
    ideal_current_frame = elapsed_time * fps
    gap_frame = current_frame - ideal_current_frame

    if gap_frame > 0:
        time.sleep(gap_frame / fps)
    elif gap_frame <= -1:
        for j in range(math.floor(abs(gap_frame))):
            current_frame += 1
            for i in range(6):
                ret, frame = videos[i].read()
                if not ret:
                    videos[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = videos[i].read()
                    current_frame = 0
                    start_time = time.perf_counter()

    current_frame += 1


# 画像のサイズをOpenGLで扱う用のサイズに変換する
def convert_size(from_img, after_size, name):
    h, w = from_img.shape[:2]
    to_img = from_img[:,w//2-h//2:w//2+h//2,:]
    to_img = cv2.resize(to_img, dsize=after_size)
    to_img = cv2.cvtColor(to_img, cv2.COLOR_BGR2RGB)
    to_img = cv2.transpose(to_img)
    to_img = cv2.flip(to_img, 0)
    to_img = cv2.rotate(to_img, cv2.ROTATE_90_CLOCKWISE)
    h, w = to_img.shape[:2]
    cv2.putText(to_img, name, (w-30-len(name)*20, h-30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
    return to_img

# 画面の更新時に呼ばれる関数
def display():
    global textures
    global texture
    global vertex
    global face
    global distance
    global r
    global ser

    # マイコンからピッチ、ヨー、ロールを取得する
    try:
        tmp_line = ''
        line = ''
        while ser.in_waiting > 0:
            val = ser.read(1)
            val = str(val).replace('b','').replace('\'','')
            if val == '\\n':
                line = tmp_line.replace('\\r','')
                tmp_line = ''
            else:
                tmp_line += val

        vals = line.split(',')
        roll, pitch, yaw = [float(val) for val in vals]

        if abs(r[2] - roll * 3) > 10:
            roll = r[2] / 3
        
        r[0] = -pitch * 3
        r[1] = (yaw-180)*5 + 180
        r[2] = roll * 3

    except Exception as e:
        print(e)

    # OpenGLの処理
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glLoadIdentity()
    gluLookAt(0.0, 0.0, distance, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    glScaled(1.5, 1.5, 1.5)
    glRotated(float(r[0]), 1.0, 0.0, 0.0)
    glRotated(float(r[1]), 0.0, 1.0, 0.0)
    glRotated(float(r[2]), 0.0, 0.0, 1.0)
    glTranslated(-0.5, -0.5, -0.5)
    glColor3f(1.0, 1.0, 1.0)
    for i in range(6):
        glBindTexture(GL_TEXTURE_2D, textures[i])
        glBegin(GL_QUADS)
        for j in range(4):
            glTexCoord2dv(texture[j])
            glVertex3dv(vertex[face[i][j]])
        glEnd()
    glDisable(GL_TEXTURE_2D)
    glutSwapBuffers()
    glFlush()


# 画面のリサイズ時に呼ばれる関数
def resize(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(30.0, float(w)/float(h), 1.0, 100.0)
    glMatrixMode(GL_MODELVIEW)


# idle()を定期的に実行するために使う
def timer(value):
    idle()
    glutTimerFunc(0, timer, 0)


if __name__ == '__main__':
    print('start program')

    # OpenGLの処理
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutCreateWindow(sys.argv[0])
    glutDisplayFunc(display)
    glutReshapeFunc(resize)

    title = sys.argv[1]

    # 最初にする処理
    init(title)

    # 動画の再生速度を合わせるために、スタート時間を記録しておく
    start_time = time.perf_counter()

    # timer関数を、定期的に実行する関数として登録する
    glutTimerFunc(0, timer, 0)

    # ループに入る
    print("start glut main loop\n")
    glutMainLoop()
