import sys

from OpenGL.WGL import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import cv2
import uuid
from win32api import *
from win32con import *
from win32gui import *
def deleteBuffer(fbo, color_buf, depth_buf, width, height ):

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteRenderbuffers(1, color_buf)
    glDeleteRenderbuffers(1, depth_buf)
    glDeleteFramebuffers(1, fbo)

def myglReadColorBuffer(fbo, color_buf, depth_buf, width, height):
    glReadBuffer(GL_COLOR_ATTACHMENT0)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)

    return data, width, height


img = cv2.imread("test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height,width,c = img.shape

PFD_TYPE_RGBA =         0
PFD_MAIN_PLANE =        0
PFD_DOUBLEBUFFER =      0x00000001
PFD_DRAW_TO_WINDOW =    0x00000004
PFD_SUPPORT_OPENGL =    0x00000020
def mywglCreateContext(hWnd):
    pfd = PIXELFORMATDESCRIPTOR()

    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL
    pfd.iPixelType = PFD_TYPE_RGBA
    pfd.cColorBits = 32
    pfd.cDepthBits = 24
    pfd.iLayerType = PFD_MAIN_PLANE

    hdc = GetDC(hWnd)

    pixelformat = ChoosePixelFormat(hdc, pfd)
    SetPixelFormat(hdc, pixelformat, pfd)

    oglrc = wglCreateContext(hdc)
    wglMakeCurrent(hdc, oglrc)

    # check is context created succesfully
    # print "OpenGL version:", glGetString(GL_VERSION)

hInstance = GetModuleHandle(None)

wndClass = WNDCLASS()

wndClass.lpfnWndProc = DefWindowProc
wndClass.hInstance = hInstance
wndClass.hbrBackground = GetStockObject(WHITE_BRUSH)
wndClass.hCursor = LoadCursor(0, IDC_ARROW)
wndClass.lpszClassName = str(uuid.uuid4())
wndClass.style = CS_OWNDC

wndClassAtom = RegisterClass(wndClass)

# don't care about window size, couse we will create independent buffers
hWnd = CreateWindow(wndClassAtom, '', WS_POPUP, 0, 0, 1, 1, 0, 0, hInstance, None)

# Ok, window created, now we can create OpenGL context

mywglCreateContext(hWnd)

fbo = glGenFramebuffers(1)
color_buf = glGenRenderbuffers(1)
depth_buf = glGenRenderbuffers(1)

glBindFramebuffer(GL_FRAMEBUFFER, fbo)



glBindRenderbuffer(GL_RENDERBUFFER, color_buf)
glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_buf)


# bind depth render buffer - no need for 2D, but necessary for real 3D rendering
glBindRenderbuffer(GL_RENDERBUFFER, depth_buf)
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buf)

#https://wikidocs.net/6442
# Although there are plenty of good imaging libraries for C/C++ I doubt that any of them are as easy to use as the Python Image Library (PIL). Although PIL's commercially licensed version has an OpenGL Image interface, with the free version in order to load an image as a texture you can do something like:

# # 's
#  1import Image
#  2img = Image.open('some_img.png') # .jpg, .bmp, etc. also work
#  3img_data = numpy.array(list(img.getdata()), numpy.int8)
#  4
#  5texture = glGenTextures(1)
#  6glPixelStorei(GL_UNPACK_ALIGNMENT,1)
#  7glBindTexture(GL_TEXTURE_2D, texture)
#  8glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
#  9glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
# 10glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# 11glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
# 12glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
# 3.6   Geometry Rendering Performance
texture = glGenTextures(1)
new_img = img.reshape(-1).astype(np.ubyte)
glBindTexture(GL_TEXTURE_2D, texture)

glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
import ctypes

glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, new_img.ctypes.data_as(ctypes.c_void_p))

# return fbo, color_buf, depth_buf, width, height


gl_program = glCreateProgram()
v_shader = glCreateShader(GL_VERTEX_SHADER)
p_shader = glCreateShader(GL_FRAGMENT_SHADER)

vertexPositions = np.array(
    [0.75, 0.75, 0.0, 1.0,
    0.75, -0.75, 0.0, 1.0, 
    -0.75, -0.75, 0.0, 1.0],
    dtype='float32'
)
v_shader_src = """
#version 330 core 

layout(location = 0) in vec4 position;
void main()
{
   gl_Position = position;
}
"""

p_shader_src = """
#version 330 core
out vec4 outputColor;
void main()
{
   outputColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
"""

glShaderSource(v_shader, v_shader_src)
glShaderSource(p_shader, p_shader_src)


glCompileShader(v_shader)
if not glGetShaderiv(v_shader, GL_COMPILE_STATUS):
    info_log = glGetShaderInfoLog(v_shader)
    print ("Compilation Failure for " + str(v_shader) + " shader:\n" + info_log)

glCompileShader(p_shader)
if not glGetShaderiv(p_shader, GL_COMPILE_STATUS):
    info_log = glGetShaderInfoLog(p_shader)
    print ("Compilation Failure for " + str(p_shader) + " shader:\n" + str(info_log))

glAttachShader(gl_program, v_shader)
glAttachShader(gl_program, p_shader)

glLinkProgram(gl_program)


# positionBufferObject = glGenBuffers(1)
    
# glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
# glBufferData( # PyOpenGL allows for the omission of the size parameter
#     GL_ARRAY_BUFFER,
#     12,
#     vertexPositions.ctypes.data_as(ctypes.c_void_p),
#     GL_STATIC_DRAW
# )
# glBindBuffer(GL_ARRAY_BUFFER, 0)

pos = glGetAttribLocation(gl_program, 'position')

###
##

glEnableVertexAttribArray(pos)


glEnable(GL_DEPTH_TEST)
glClearColor(0.0,0.0,0.0,1.0)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

glUseProgram(gl_program)
# glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
# glEnableVertexAttribArray(0)
# glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
glVertexAttribPointer(pos, 4, GL_FLOAT, GL_FALSE, 0, vertexPositions.ctypes.data_as(ctypes.c_void_p))
print(glGetError())
glDrawArrays(GL_TRIANGLES, 0, 3)
print(glGetError())

glDisableVertexAttribArray(0)
glUseProgram(0)
print(glGetError())


data, _, _ = myglReadColorBuffer(fbo, color_buf, depth_buf, width, height)
cv2.imshow("test", data)












