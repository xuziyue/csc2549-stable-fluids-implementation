import numpy as np
import taichi as ti
import cv2
import argparse
import time
import random
# use CPU
ti.init(arch=ti.x64)

# Parameters
# Resolution
res = 256
# Time step
dt = 0.03
# force strength from mouse dragging
force_strength = 10000.0
# fluid viscosity for diffusion
viscosity = 0.9
diffusion_rate = 0.9
# fluid dissipate rate for dissipation
dissipate_rate = 0.1
# whether to use sparse method, considering the sparsity, this should be True
use_sparse_matrix = True

color_i = 0
# create some colors for the fluids in the standard mode
colors = [np.array([1.0, 0.0, 0.0]),
          np.array([0.0, 1.0, 0.0]),
          np.array([0.0, 0.0, 1.0]),
          np.array([1.0, 0.7, 0.0]),
          np.array([0.6, 0.0, 0.8]),
          np.array([1.0, 0.5, 0.0])]
color = colors[color_i]

# Getting arguments of modes and boundary methods
parser = argparse.ArgumentParser()
parser.add_argument('--standard', dest='standard', action='store_true', help='create fluids and interaction')
parser.add_argument('--smoke', dest='smoke', action='store_true', help='show smoke simulation effect')
parser.add_argument('--picture', dest='picture', action='store_true', help='show a picture and create fluids effect based on the picture')
parser.add_argument('--periodic_boundary', dest='periodic_boundary', action='store_true', help='periodic boundary method')
parser.add_argument('--fixed_boundary', dest='fixed_boundary', action='store_true', help='fixed boundary method')
args = parser.parse_args()

# Setting mode and boundary method according to arguments
mode = 'standard'
boundary_method = 'fixed'  # default fixed
if args.smoke:
    mode = 'smoke'
    boundary_method = 'fixed'
    viscosity = 0.001
    diffusion_rate = 0.0000001
    dissipate_rate = 0.001
elif args.picture:
    mode = 'picture'
    boundary_method = 'fixed'
elif args.standard:
    mode = 'standard'
    boundary_method = 'fixed'

if args.periodic_boundary:
    boundary_method = 'periodic'  # periodic
elif args.fixed_boundary:
    boundary_method = 'fixed'     # fixed

print("You are playing the {} mode.".format(mode))

# Global placeholder for velocity, density, and etc.
# velocity
U0 = ti.Vector.field(2, float, shape=(res, res))
U1 = ti.Vector.field(2, float, shape=(res, res))
# density
S0 = ti.Vector.field(3, float, shape=(res, res))
S1 = ti.Vector.field(3, float, shape=(res, res))
q = ti.field(float, shape=(res, res))
# velocity
forces = ti.Vector.field(2, float, shape=(res, res))
# sources
sources = ti.Vector.field(3, float, shape=(res, res))
# velocity derivatives
velocity_derivatives = ti.field(float, shape=(res, res))


# Create the sparse taichi discrete laplace operator using periodic boundary method
# https://docs.taichi.graphics/lang/articles/advanced/sparse_matrix
@ti.kernel
def discrete_laplace_operator_periodic(M: ti.linalg.sparse_matrix_builder()):
    for block_i in range(res):
        for block_j in range(res):
            if block_i == block_j:
                for i in range(block_i * res, block_i * res + res):
                    M[i, i] += 4
                    if 0 <= i <= (res - 1):
                        M[i, i + res * (res - 1)] += -1
                    if res * (res - 1) < i < (res + 1) * (res - 1):
                        M[i, i - res * (res - 1)] += -1
                    if i % res == 0:
                        M[i, i + (res - 1)] += -1
                    if i % res == (res - 1):
                        M[i, i - (res - 1)] += -1

                    if i == block_i * res:
                        M[i, i + 1] += -1
                    elif i == block_i * res + res - 1:
                        M[i, i - 1] += -1
                    else:
                        M[i, i + 1] += -1
                        M[i, i - 1] += -1
            elif block_j == block_i - 1 or block_j == block_i + 1:
                for i in range(res):
                    M[block_i * res + i, block_j * res + i] += -1


# Create the sparse taichi discrete laplace operator using fixed boundary method
# https://docs.taichi.graphics/lang/articles/advanced/sparse_matrix
@ti.kernel
def discrete_laplace_operator_fixed(M: ti.linalg.sparse_matrix_builder()):
    # for each block od size res * res
    for block_i in range(res):
        for block_j in range(res):
            if block_i == block_j:
                for i in range(block_i * res, block_i * res + res):
                    if i == 0 or i == (res - 1) or i == res * (res - 1) or i == (res + 1) * (res - 1):
                        M[i, i] += 2
                    elif 0 < i < (res - 1) or i % res == 0 or i % res == (res - 1) or res * (res - 1) < i < (
                            res + 1) * (res - 1):
                        M[i, i] += 3
                    else:
                        M[i, i] += 4
                    if i == block_i * res:
                        M[i, i + 1] += -1
                    elif i == block_i * res + res - 1:
                        M[i, i - 1] += -1
                    else:
                        M[i, i + 1] += -1
                        M[i, i - 1] += -1
            elif block_j == block_i - 1 or block_j == block_i + 1:
                for i in range(res):
                    M[block_i * res + i, block_j * res + i] += -1


# Create the sparse taichi identity matrix
@ti.kernel
def identity_matrix(I: ti.linalg.sparse_matrix_builder()):
    for i in range(res * res):
        I[i, i] += 1


# Create the sparse taichi discrete laplace operator A
A = ti.linalg.SparseMatrixBuilder(res * res, res * res, max_num_triplets=res * res * res)
if boundary_method == 'fixed':
    discrete_laplace_operator_fixed(A)
elif boundary_method == 'periodic':
    discrete_laplace_operator_periodic(A)
else:
    print('boundary_method wrong')
A = A.build()

# Create the sparse taichi identity matrix I and sparse taichi diffuse matrix
I = ti.linalg.SparseMatrixBuilder(res * res, res * res, max_num_triplets=res * res * res)
identity_matrix(I)
I = I.build()
# I - nu * dt * laplace operator
diffuse_matrix_velocity = I + viscosity * dt * A
diffuse_matrix_density = I + diffusion_rate * dt * A

# Create the place holder of b for solving A x = b later
b = ti.field(ti.f32, shape=res*res)
b_1 = ti.field(ti.f32, shape=res*res)
b_2 = ti.field(ti.f32, shape=res*res)
b_3 = ti.field(ti.f32, shape=res*res)

# Create the solver for A x = b for projection
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(A)
solver.factorize(A)

# Create the solver for diffuse_matrix x = b for diffusion
solver_velocity = ti.linalg.SparseSolver(solver_type="LLT")
solver_velocity.analyze_pattern(diffuse_matrix_velocity)
solver_velocity.factorize(diffuse_matrix_velocity)


# Create the solver for diffuse_matrix x = b for diffusion
solver_density = ti.linalg.SparseSolver(solver_type="LLT")
solver_density.analyze_pattern(diffuse_matrix_density)
solver_density.factorize(diffuse_matrix_density)


# =======================================General function========================================
# Linear interpolation
@ti.func
def lin_interp(vf, p):
    # https://en.wikipedia.org/wiki/Bilinear_interpolation
    # https://docs.taichi.graphics/lang/articles/basic/field
    x, y = p
    x, y = x - 0.5, y - 0.5
    if x < 0:
        x = 0
    if x > res - 1:
        x = res - 1
    if y < 0:
        y = 0
    if y > res - 1:
        y = res - 1
    
    x_1 = int(ti.floor(x))
    y_1 = int(ti.floor(y))
    x_2 = int(ti.ceil(x))
    y_2 = int(ti.ceil(y))

    Q11 = vf[x_1, y_1]
    Q21 = vf[x_2, y_1]
    Q12 = vf[x_1, y_2]
    Q22 = vf[x_2, y_2]
    R1 = ti.Vector([0.0] * Q11.n)
    R2 = ti.Vector([0.0] * Q11.n)
    P = ti.Vector([0.0] * Q11.n)
    if x_1 == x_2:
        R1 = Q11
        R2 = Q12
    else:
        R1 = (x_2 - x) / (x_2 - x_1) * Q11 + (x - x_1) / (x_2 - x_1) * Q21
        R2 = (x_2 - x) / (x_2 - x_1) * Q12 + (x - x_1) / (x_2 - x_1) * Q22
    if y_1 == y_2:
        P = R1
    else:
        P = (y_2 - y) / (y_2 - y_1) * R1 + (y - y_1) / (y_2 - y_1) * R2
    return P


# 3rd order Runge-Kutta
@ti.func
def trace_particle(vf, p):
    v1 = lin_interp(vf, p) # p is [i + 0.5, j + 0.5]
    return p - 0.5 * dt * v1


@ti.kernel
def transport(new_qf: ti.template(), qf: ti.template(), vf: ti.template()):
    for i, j in qf:
        x = ti.Vector([i + 0.5, j + 0.5])
        x_0 = trace_particle(vf, x) # find the position dt times ago
        new_qf[i, j] = lin_interp(qf, x_0) # new_qf[i, j] is set to be qf at p


# Apply force on velocity
@ti.kernel
def apply_force(vf: ti.template(), forces: ti.template()):
    for i, j in vf:
        vf[i, j] += dt * forces[i, j]


# Apply source on density
@ti.kernel
def apply_source(dyef: ti.template(), sources: ti.template()):
    for i, j in dyef:
        dyef[i, j] += dt * sources[i, j]


# Update
@ti.kernel
def update(vf: ti.template(), dyef: ti.template(), forces: ti.template(), sources: ti.template()):
    apply_force(vf, forces)
    apply_source(dyef, sources)


# Clear vector, can clear sources, forces, velocities, and densities
@ti.kernel
def clear_vector(vector: ti.template()):
    for i, j in vector:
        vector[i, j].fill(0)


# Reset sources, forces, velocities, and densities
def reset():
    print('reset')
    clear_vector(U0)
    clear_vector(S0)
    clear_vector(U1)
    clear_vector(S1)
    clear_vector(sources)
    clear_vector(forces)
    if mode == 'picture':
        add_sources_picture(S1, img)
# =======================================Standard mode===========================================
# Apply impulse (add dye and forces) from mouse dragging in standard mode
@ti.kernel
def apply_impulse(dyef: ti.template(), imp_data: ti.ext_arr(), forces: ti.template()):
    mdir = ti.Vector([imp_data[0], imp_data[1]]) # mouse direction
    mouse_x, mouse_y = max(0, min(int(imp_data[2]), res-1)), max(0, min(int(imp_data[3]), res-1))
    forces[mouse_x, mouse_y] += mdir * force_strength
    dyef[mouse_x, mouse_y] += 100 * ti.Vector([imp_data[4], imp_data[5], imp_data[6]])


# =======================================Smoke mode==============================================
# Add constance smoke source
@ti.kernel
def add_sources_smoke(sources: ti.template()):
    smoke_size = 10
    smoke_strength = 3
    smoke_x, smoke_y = 128, 56
    for i, j in sources:
        if (i-smoke_x) ** 2 + (j-smoke_y) ** 2 <= smoke_size:
            sources[i, j] = ti.Vector([1, 1, 1]) * smoke_strength


@ti.kernel
def add_force_smoke_buoyancy(forces: ti.template(), random_strength: float):
    smoke_x, smoke_y = 128, 56
    decay_rate = 0.01

    for i, j in forces:
        distance = (i - smoke_x) ** 2
        factor = ti.exp(-distance * decay_rate)
        forces[i, j] += ti.Vector([0, 0.001]) * force_strength * random_strength * factor


@ti.kernel
def add_force_smoke_wind(forces: ti.template(), random_x: float, random_strength: float):
    for i, j in forces:
        if 50 <= i <= res - 51 and 50 <= j <= res - 51:
            forces[i, j] += ti.Vector([random_x, 0]) * force_strength * random_strength


# =======================================Picture mode============================================
@ti.kernel
def add_sources_picture(sources: ti.template(), img: ti.ext_arr()):
    for i, j in sources:
        sources[i, j] = ti.Vector([img[i, j, 0] / 255, img[i, j, 1] / 255, img[i, j, 2] / 255])


@ti.kernel
def add_force_picture(imp_data: ti.ext_arr(), forces: ti.template()):
    force_size = 5
    for i, j in forces:
        if (i - int(imp_data[2])) ** 2 + (j - int(imp_data[3])) ** 2 <= force_size:
            mdir = ti.Vector([imp_data[0], imp_data[1]])  # mouse direction
            forces[i, j] += mdir * force_strength * 1

@ti.kernel
def add_force_smoke(imp_data: ti.ext_arr(), forces: ti.template()):
    mdir = ti.Vector([imp_data[0], imp_data[1]])
    mouse_x, mouse_y = max(0, min(int(imp_data[2]), res - 1)), max(0, min(int(imp_data[3]), res - 1))
    forces[mouse_x, mouse_y] += mdir * force_strength * 10


# Define how to interact from mouse data under different modes
def interactive(mouse_data: ti.ext_arr(), forces: ti.template(), sources: ti.template()):
    if mode == 'standard':
        if gui.is_pressed(ti.GUI.LMB):
            # Add dye and forces when mouse is pressed
            apply_impulse(S0, mouse_data, forces)
        else:
            # Clear forces when mouse is not pressed
            clear_vector(forces)

    elif mode == 'smoke':
        clear_vector(forces)
        # Add sources of smoke
        add_sources_smoke(sources)

        # Add buoyancy and random wind
        random_strength = (random.random() / 2 + 0.5)
        add_force_smoke_buoyancy(forces, random_strength)

        if ((time.time() // 5) % 2) == 0:
            random_x = random.random() * 0.0001
        else:
            random_x = -random.random() * 0.0001
        add_force_smoke_wind(forces, random_x, 1)

        if gui.is_pressed(ti.GUI.LMB):
            # Add forces when mouse is pressed
            add_force_smoke(mouse_data, forces)

    elif mode == 'picture':
        if gui.is_pressed(ti.GUI.LMB):
            # Add forces when mouse is pressed
            add_force_picture(mouse_data, forces)
        else:
            clear_vector(forces)


# =======================================Key functions for Vstep and Sstep=======================
@ti.kernel
def divergence_periodic(w: ti.template()):
    for i, j in w:
        if i == 0:
            if j == 0:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[res - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, res - 1][1]) / 2
            elif j == res - 1:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[res - 1, j][0]) / 2 + (w[i, 0][1] - w[i, j - 1][1]) / 2
            else:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[res - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2
        elif i == res - 1:
            if j == 0:
                velocity_derivatives[i, j] = (w[0, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, res - 1][1]) / 2
            elif j == res - 1:
                velocity_derivatives[i, j] = (w[0, j][0] - w[i - 1, j][0]) / 2 + (w[i, 0][1] - w[i, j - 1][1]) / 2
            else:
                velocity_derivatives[i, j] = (w[0, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2
        else:
            if j == 0:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, res - 1][1]) / 2
            elif j == res - 1:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, 0][1] - w[i, j - 1][1]) / 2
            else:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2


@ti.kernel
def divergence_fixed(w: ti.template()):
    for i, j in w:
        if i == 0:
            if j == 0:
                velocity_derivatives[i, j] = (w[i + 1, j][0] + w[i, j][0]) / 2 + (w[i, j + 1][1] + w[i, j][1]) / 2
            elif j == res - 1:
                velocity_derivatives[i, j] = (w[i + 1, j][0] + w[i, j][0]) / 2 + (-w[i, j][1] - w[i, j - 1][1]) / 2
            else:
                velocity_derivatives[i, j] = (w[i + 1, j][0] + w[i, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2
        elif i == res - 1:
            if j == 0:
                velocity_derivatives[i, j] = (-w[i, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] + w[i, j][1]) / 2
            elif j == res - 1:
                velocity_derivatives[i, j] = (-w[i, j][0] - w[i - 1, j][0]) / 2 + (-w[i, j][1] - w[i, j - 1][1]) / 2
            else:
                velocity_derivatives[i, j] = (-w[i, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2
        else:
            if j == 0:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] + w[i, j][1]) / 2
            elif j == res - 1:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (-w[i, j][1] - w[i, j - 1][1]) / 2
            else:
                velocity_derivatives[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        left = i - 1
        right = i + 1
        up = j + 1
        down = j - 1
        if left < 0:
            left = 0
        if right > res - 1:
            right = res - 1
        if up > res - 1:
            up = res - 1
        if down < 0:
            down = 0
        dqdx = (pf[right, j] - pf[left, j]) / 2
        dqdy = (pf[i, up] - pf[i, down]) / 2
        vf[i, j] -= ti.Vector([dqdx, dqdy])


@ti.kernel
def copy_divergence(div_in: ti.template(), div_out: ti.template()):
    for i, j in div_in: # I = i, j
        div_out[i * res + j] = -div_in[i, j] # div_out[i * res + j] = -velocity_divs[i, j]


@ti.kernel
def flatten_velocity(div_in: ti.template(),
                     div_out_1: ti.template(),
                     div_out_2: ti.template()):
    for i, j in div_in: # I = i, j
        div_out_1[i + j * res] = div_in[i, j][0] # div_out[i * res + j] = -velocity_divs[i, j]
        div_out_2[i + j * res] = div_in[i, j][1]


@ti.kernel
def flatten_density(div_in: ti.template(),
                    div_out_1: ti.template(),
                    div_out_2: ti.template(),
                    div_out_3: ti.template()):
    for i, j in div_in: # I = i, j
        div_out_1[i + j * res] = div_in[i, j][0] # div_out[i * res + j] = -velocity_divs[i, j]
        div_out_2[i + j * res] = div_in[i, j][1]
        div_out_3[i + j * res] = div_in[i, j][2]


@ti.kernel
def apply_pressure(p_in: ti.ext_arr(), p_out: ti.template()):
    for i, j in p_out:
        p_out[i, j] = p_in[i * res + j]


@ti.kernel
def add_to_velocity(p_in_1: ti.ext_arr(), p_in_2: ti.ext_arr(), p_out: ti.template()):
    for i, j in p_out:
        p_out[i, j][0] = p_in_1[i + j * res]
        p_out[i, j][1] = p_in_2[i + j * res]


@ti.kernel
def add_to_density(p_in_1: ti.ext_arr(),
                   p_in_2: ti.ext_arr(),
                   p_in_3: ti.ext_arr(),
                   p_out: ti.template()):
    for i, j in p_out:
        p_out[i, j][0] = p_in_1[i + j * res]
        p_out[i, j][1] = p_in_2[i + j * res]
        p_out[i, j][2] = p_in_3[i + j * res]


def diffuse_velocity(vf):
    flatten_velocity(vf, b_1, b_2)
    x_1 = solver_velocity.solve(b_1)
    x_2 = solver_velocity.solve(b_2)
    add_to_velocity(x_1, x_2, vf)


def diffuse_density(vf):
    flatten_density(vf, b_1, b_2, b_3)
    x_1 = solver_density.solve(b_1)
    x_2 = solver_density.solve(b_2)
    x_3 = solver_density.solve(b_3)
    add_to_density(x_1, x_2, x_3, vf)


def project(vf):
    if boundary_method == 'fixed':
        divergence_fixed(vf)
    elif boundary_method == 'periodic':
        divergence_periodic(vf)
    else:
        print('boundary_method wrong')
    copy_divergence(velocity_derivatives, b)
    x = solver.solve(b)
    apply_pressure(x, q)
    subtract_gradient(vf, q)


@ti.kernel
def dissipate(df: ti.template()):
    for i, j in df:
        df[i, j] = df[i, j] / (1 + dt * dissipate_rate)


def Vstep():
    apply_force(U0, forces)
    
    transport(U1, U0, U0)
    diffuse_velocity(U1)
    project(U1)
    dissipate(U1)


def Sstep():
    apply_source(S0, sources)
    
    transport(S1, S0, U1)
    if mode != 'picture':
        diffuse_density(S1)
        dissipate(S1)

# =======================================Pre-processing==========================================
cursor = None
direction = None

if mode == 'picture':
    img = cv2.imread('latte.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (res, res))

    viscosity = 0.0
    add_sources_picture(S1, img)

# =======================================GUI=====================================================
gui = ti.GUI('Stable Fluid', (res, res))
while gui.running:
    # handle user interaction
    if gui.get_event((ti.GUI.PRESS, ti.GUI.ESCAPE)):
        gui.running = False

    if gui.is_pressed(ti.GUI.LEFT, 'r'):
        reset()

    mouse_data = np.zeros(8, dtype=np.float32)
    if gui.is_pressed(ti.GUI.LMB):
        mouse_x, mouse_y = gui.get_cursor_pos()
        mouse_x *= res
        mouse_y *= res

        mouse_x = min(mouse_x, res - 1)
        mouse_y = min(mouse_y, res - 1)
        mouse_x = max(mouse_x, 0)
        mouse_y = max(mouse_y, 0)

        if cursor:
            direction = mouse_x - cursor[0], mouse_y - cursor[1]
            mouse_data[0], mouse_data[1] = direction[0], direction[1]
            mouse_data[2], mouse_data[3] = mouse_x, mouse_y
            mouse_data[4:7] = color

        cursor = mouse_x, mouse_y

    else:
        cursor = None
        direction = None
        color_i += 1
        color = colors[color_i % len(colors)]

    U0, U1 = U1, U0
    S0, S1 = S1, S0

    interactive(mouse_data, forces, sources)

    Vstep()
    Sstep()
    gui.set_image(S0)
    gui.show()
