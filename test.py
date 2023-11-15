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

import glm
import igl
v, f = igl.read_triangle_mesh("identity090.obj")
v, f = igl.read_triangle_mesh("tea.obj")
mean = np.mean(v, axis=0)

v, f = igl.read_triangle_mesh("cube.obj")
# v = np.array([[-0.5, 0.0, 0.0],[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
# f = np.array([[2,1,0],[0,3,2]], dtype=np.uint)




vec1 = v[f[:, 1]] - v[f[:, 0]]
vec2 = v[f[:, 2]] - v[f[:, 0]]
v_normal = np.zeros_like(v)
v_normal_denorm = np.zeros((len(v), 1))
f_normal = np.cross(vec1, vec2, axis=-1)
for fi, (fn, (vi1,vi2,vi3)) in  enumerate(zip(f_normal, f)):
    v_normal[vi1] += fn
    v_normal[vi2] += fn
    v_normal[vi3] += fn
    v_normal_denorm[vi1] += 1
    v_normal_denorm[vi2] += 1
    v_normal_denorm[vi3] += 1
v_normal /= v_normal_denorm

# v = (v - mean)
# v[:, -1] -= 2
vv = np.concatenate([v, np.ones((v.shape[0], 1))], axis= -1)


def deleteBuffer(fbo, color_buf, depth_buf, width, height ):

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteRenderbuffers(1, color_buf)
    glDeleteRenderbuffers(1, depth_buf)
    glDeleteFramebuffers(1, fbo)

def myglReadColorBuffer(fbo, color_buf, depth_buf, width, height):
    glReadBuffer(GL_COLOR_ATTACHMENT0)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    res = np.zeros((width*height*4), dtype=np.uint8)
    
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, res.ctypes.data_as(ctypes.c_void_p))
    res = res.reshape(height, width, -1)
    return res, width, height


img = cv2.imread("test.jpg")
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

v_shader_src = """
#version 330 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec3 anormal;

uniform mat4 proj;
uniform mat4 mvp;

out vec3 FragPos;
out vec3 Normal;

void main()
{

    gl_Position = proj*mvp*aPos; 
    //FragPos = vec3( mvp * aPos);
    //Normal = anormal;
}
"""

p_shader_src = """
#version 330 core
out vec4 FragColor;
uniform vec3 lightColor;
uniform vec3 objectColor;

uniform vec3 lightPos;

in vec3 Normal;

in vec3 FragPos;


void main()
{
    // float ambientStrength = 0.1;
    // vec3 ambient = ambientStrength * (lightColor);
    // vec3 norm = normalize(Normal);
    // vec3 lightDir = normalize(lightPos - FragPos);
    // float diff = max(dot(norm, lightDir), 0.0);
    // vec3 diffuse = diff*lightColor;
    // vec3 result = (ambient + diffuse)*objectColor;
    // FragColor = vec4(result, 1.0);
    FragColor = vec4(lightColor * objectColor, 1.0);
}
"""

glShaderSource(v_shader, v_shader_src)
glShaderSource(p_shader, p_shader_src)


glCompileShader(v_shader)
if not glGetShaderiv(v_shader, GL_COMPILE_STATUS):
    info_log = glGetShaderInfoLog(v_shader)
    print ("vertex shader Compilation Failure for " + str(v_shader) + " shader:\n" + str(info_log))

glCompileShader(p_shader)
if not glGetShaderiv(p_shader, GL_COMPILE_STATUS):
    info_log = glGetShaderInfoLog(p_shader)
    print ("pixel shader Compilation Failure for " + str(p_shader) + " shader:\n" + str(info_log))
    print(p_shader_src)
glAttachShader(gl_program, v_shader)
glAttachShader(gl_program, p_shader)

glLinkProgram(gl_program)
glDeleteShader(v_shader)
glDeleteShader(p_shader)    


status = glGetProgramiv(gl_program, GL_LINK_STATUS)
if status :
    test = glGetProgramInfoLog(gl_program)
    print(test)

vv = np.concatenate([vv, v_normal], axis=-1)
vv = vv.astype(np.float32).reshape(-1)


vertexPositions = np.zeros_like(vv, dtype=np.float32)

vertexPositions[...] = vv
# vertexPositions = np.array(
    # [-0.5, -0.5, 0.0, 1.0,
    # 0.5, -0.5, 0.0, 1.0, 
    # 0.0, 0.5, 0.0, 1.0],
    # dtype='float32'
# )
# f = np.array([0,1,2], dtype=np.uint32)
f = f.astype(np.uint32).reshape(-1)
ff = np.zeros_like(f, dtype=np.uint32)
ff[...] = f
f = ff
vao = glGenVertexArrays(1)
positionBufferObject = glGenBuffers(1)
# glBindVertexArray(vao)
print(vertexPositions.data.contiguous)
glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
glBufferData(GL_ARRAY_BUFFER, vertexPositions.itemsize*vertexPositions.size, vertexPositions.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)


IndexBufferObject = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferObject)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, f.itemsize*f.size, f.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)
# glBufferData(GL_ELEMENT_ARRAY_BUFFER, index.itemsize*index.size, index.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)


trans = glm.translate(glm.vec3(0, 0, -1))
proj = glm.frustum(-2, 2, 2, -2, 0.1, 1000)
# glBufferData(GL_ARRAY_BUFFER, 48, vertexPositions.ctypes.data_as(ctypes.c_void_p), GL_DYNAMIC_DRAW)

proj_loc = glGetUniformLocation(gl_program, 'proj')
mvp_loc = glGetUniformLocation(gl_program, 'mvp')
obj_color_loc = glGetUniformLocation(gl_program, 'objectColor')
light_loc = glGetUniformLocation(gl_program, 'lightColor')
light_pos_loc = glGetUniformLocation(gl_program, 'lightPos')

glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertexPositions.itemsize*7, ctypes.c_void_p(0))
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertexPositions.itemsize*7, ctypes.c_void_p(4))
glEnableVertexAttribArray(0)


# glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)



###
##
ir = 0
glViewport(0,0,width,height)
######################################################
glEnable(GL_DEPTH_TEST)
glClearColor(0.0,0.0,0.0,0.0)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)



glUseProgram(gl_program)


color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
light = np.array([1.0, 1.0, 1.0], dtype=np.float32)
lightPos = np.array([10,0, 0.0, 1.0], dtype=np.float32)

print(np.array(trans))
print(np.array(proj))
glUniformMatrix4fv(proj_loc, 1, GL_FALSE,  glm.value_ptr(proj))
glUniformMatrix4fv(mvp_loc, 1, GL_FALSE,  glm.value_ptr(trans))
glUniform3f(light_loc, *color.ravel())
glUniform3f(obj_color_loc, *light.ravel())
glUniform3f(light_pos_loc, *light.ravel())



# glBindVertexArray(vao)
# glEnableVertexAttribArray(0)
# glDrawArrays(GL_TRIANGLES, 0, 3)
glDrawElements(GL_TRIANGLES, len(f)//3, GL_UNSIGNED_INT, ctypes.c_void_p(0))
# glDisableVertexAttribArray(0)
# glDisableVertexAttribArray(pos)
glUseProgram(0)

ir += 0.2

# res = np.zeros_like(vertexPositions, dtype=np.float32)
# glGetBufferSubData(GL_ARRAY_BUFFER , 0, vertexPositions.itemsize*vertexPositions.size, res.ctypes.data_as(ctypes.c_void_p))
# print(res)
# res = np.zeros_like(ff, dtype=np.uint)
# glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER , 0, res.itemsize*res.size, res.ctypes.data_as(ctypes.c_void_p))
# print(res)
data, _, _ = myglReadColorBuffer(fbo, color_buf, depth_buf, width, height)
cv2.imshow("test", data)
cv2.waitKey(0)



# while True:
#     glEnable(GL_DEPTH_TEST)
#     glClearColor(0.0,0.0,0.0,0.0)
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)



#     glUseProgram(gl_program)
   

#     color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
#     light = np.array([1.0, 1.0, 1.0], dtype=np.float32)
#     lightPos = np.array([10,0, 0.0, 1.0], dtype=np.float32)


#     glUniformMatrix4fv(proj_loc, 1, GL_FALSE,  glm.value_ptr(proj))
#     glUniformMatrix4fv(mvp_loc, 1, GL_FALSE,  glm.value_ptr(trans))
#     glUniform3f(light_loc, *color.ravel())
#     glUniform3f(obj_color_loc, *light.ravel())
#     glUniform3f(light_pos_loc, *light.ravel())



#     # glBindVertexArray(vao)
#     # glEnableVertexAttribArray(0)
#     # glDrawArrays(GL_TRIANGLES, 0, 3)
#     glDrawElements(GL_TRIANGLES, len(f)//3, GL_UNSIGNED_INT, ctypes.c_void_p(0))
#     # glDisableVertexAttribArray(0)
#     # glDisableVertexAttribArray(pos)
#     glUseProgram(0)

#     ir += 0.2

#     # res = np.zeros_like(vertexPositions, dtype=np.float32)
#     # glGetBufferSubData(GL_ARRAY_BUFFER , 0, vertexPositions.itemsize*vertexPositions.size, res.ctypes.data_as(ctypes.c_void_p))
#     # print(res)
#     # res = np.zeros_like(ff, dtype=np.uint)
#     # glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER , 0, res.itemsize*res.size, res.ctypes.data_as(ctypes.c_void_p))
#     # print(res)
#     data, _, _ = myglReadColorBuffer(fbo, color_buf, depth_buf, width, height)
#     cv2.imshow("test", data)
#     cv2.waitKey(100)












