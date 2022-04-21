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
dt = 0.02
# fluid viscosity rate
viscosity_rate = 0.9
# fluid diffusion rate
diffusion_rate = 0.9
# fluid dissipation rate
dissipation_rate = 0.1

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
    viscosity_rate = 0.001
    diffusion_rate = 0.0000001
    dissipation_rate = 0.001
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
# Doing this because Taichi cannot return a new field. Taichi can only modify a field.
# velocity field
U0 = ti.Vector.field(2, float, shape=(res, res))
U1 = ti.Vector.field(2, float, shape=(res, res))
# density field
S0 = ti.Vector.field(3, float, shape=(res, res))
S1 = ti.Vector.field(3, float, shape=(res, res))
# externel forces
forces = ti.Vector.field(2, float, shape=(res, res))
# sources
sources = ti.Vector.field(3, float, shape=(res, res))

# =======================================General function========================================
# Linear interpolation
@ti.func
def lin_interp(velocity_field, point):
    # https://en.wikipedia.org/wiki/Bilinear_interpolation
    # https://docs.taichi.graphics/lang/articles/basic/field
    x, y = point
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

    Q11 = velocity_field[x_1, y_1]
    Q21 = velocity_field[x_2, y_1]
    Q12 = velocity_field[x_1, y_2]
    Q22 = velocity_field[x_2, y_2]
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


# Clear vector, can clear sources, forces, velocities, and densities
@ti.kernel
def clear_vector(vector_field: ti.template()):
    for i, j in vector_field:
        vector_field[i, j].fill(0)


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

# force and density strengths from mouse dragging
force_strength = 10000.0
density_strength = 100.0

# =======================================Standard mode===========================================
# Add forces and sources from mouse dragging in standard mode
@ti.kernel
def add_forces_and_sources(density_field: ti.template(), user_inputs: ti.ext_arr()):
    mdir = ti.Vector([user_inputs[0], user_inputs[1]]) # mouse direction
    mouse_x, mouse_y = max(0, min(int(user_inputs[2]), res-1)), max(0, min(int(user_inputs[3]), res-1))
    forces[mouse_x, mouse_y] += mdir * force_strength
    density_field[mouse_x, mouse_y] += density_strength * ti.Vector([user_inputs[4], user_inputs[5], user_inputs[6]])


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


@ti.kernel
def add_force_smoke(user_inputs: ti.ext_arr(), forces: ti.template()):
    mdir = ti.Vector([user_inputs[0], user_inputs[1]])
    mouse_x, mouse_y = max(0, min(int(user_inputs[2]), res - 1)), max(0, min(int(user_inputs[3]), res - 1))
    forces[mouse_x, mouse_y] += mdir * force_strength * 10

# =======================================Picture mode============================================
@ti.kernel
def add_sources_picture(sources: ti.template(), img: ti.ext_arr()):
    for i, j in sources:
        sources[i, j] = ti.Vector([img[i, j, 0] / 255, img[i, j, 1] / 255, img[i, j, 2] / 255])


@ti.kernel
def add_force_picture(user_inputs: ti.ext_arr(), forces: ti.template()):
    force_size = 5
    for i, j in forces:
        if (i - int(user_inputs[2])) ** 2 + (j - int(user_inputs[3])) ** 2 <= force_size:
            mdir = ti.Vector([user_inputs[0], user_inputs[1]])  # mouse direction
            forces[i, j] += mdir * force_strength


# Define how to interact from mouse data under different modes
def interactive(user_inputs: ti.ext_arr(), forces: ti.template(), sources: ti.template()):
    if mode == 'standard':
        if gui.is_pressed(ti.GUI.LMB):
            # Add sources and forces when mouse is pressed
            add_forces_and_sources(S0, user_inputs)
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
            add_force_smoke(user_inputs, forces)

    elif mode == 'picture':
        if gui.is_pressed(ti.GUI.LMB):
            # Add forces when mouse is pressed
            add_force_picture(user_inputs, forces)
        else:
            clear_vector(forces)


# =======================================Key functions for Vstep and Sstep=======================

# Update velocity field due to the change of forces
@ti.kernel
def add_force(velocity_field: ti.template(), forces: ti.template()):
    for i, j in velocity_field:
        velocity_field[i, j] += dt * forces[i, j]


# Update density field due to the change of sources
@ti.kernel
def add_source(density_field: ti.template(), sources: ti.template()):
    for i, j in density_field:
        density_field[i, j] += dt * sources[i, j]


# Advection
@ti.kernel
def transport(new_field: ti.template(), current_field: ti.template(), velocity_field: ti.template()):
    for i, j in current_field:
        x = ti.Vector([float(i), float(j)])
        # backtrace method
        v = lin_interp(velocity_field, x)
        x_0 = x - dt * v
        new_field[i, j] = lin_interp(current_field, x_0)


# Diffusion and projection preparations

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
diffuse_matrix_velocity = I + viscosity_rate * dt * A
# I - kappa * dt * laplace operator
diffuse_matrix_density = I + diffusion_rate * dt * A

# Create a sparse matrix solver for A u = b for projection
poisson_equation_solver = ti.linalg.SparseSolver(solver_type="LLT")
poisson_equation_solver.analyze_pattern(A)
poisson_equation_solver.factorize(A)

# Create a sparse matrix solver for diffuse_matrix w_new = w_old for diffusion
solver_velocity = ti.linalg.SparseSolver(solver_type="LLT")
solver_velocity.analyze_pattern(diffuse_matrix_velocity)
solver_velocity.factorize(diffuse_matrix_velocity)

# Create a sparse matrix solver for diffuse_matrix w_new = w_old for diffusion
solver_density = ti.linalg.SparseSolver(solver_type="LLT")
solver_density.analyze_pattern(diffuse_matrix_density)
solver_density.factorize(diffuse_matrix_density)


# compute velocity filed divergence
velocity_field_divergence = ti.field(float, shape=(res, res))
# velocity filed divergence for periodic boundary
@ti.kernel
def divergence_periodic(w: ti.template()):
    for i, j in w:
        if i == 0:
            if j == 0:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[res - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, res - 1][1]) / 2
            elif j == res - 1:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[res - 1, j][0]) / 2 + (w[i, 0][1] - w[i, j - 1][1]) / 2
            else:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[res - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2
        elif i == res - 1:
            if j == 0:
                velocity_field_divergence[i, j] = (w[0, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, res - 1][1]) / 2
            elif j == res - 1:
                velocity_field_divergence[i, j] = (w[0, j][0] - w[i - 1, j][0]) / 2 + (w[i, 0][1] - w[i, j - 1][1]) / 2
            else:
                velocity_field_divergence[i, j] = (w[0, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2
        else:
            if j == 0:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, res - 1][1]) / 2
            elif j == res - 1:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, 0][1] - w[i, j - 1][1]) / 2
            else:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2


# velocity filed divergence for fixed boundary
@ti.kernel
def divergence_fixed(w: ti.template()):
    for i, j in w:
        if i == 0:
            if j == 0:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] + w[i, j][0]) / 2 + (w[i, j + 1][1] + w[i, j][1]) / 2
            elif j == res - 1:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] + w[i, j][0]) / 2 + (-w[i, j][1] - w[i, j - 1][1]) / 2
            else:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] + w[i, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2
        elif i == res - 1:
            if j == 0:
                velocity_field_divergence[i, j] = (-w[i, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] + w[i, j][1]) / 2
            elif j == res - 1:
                velocity_field_divergence[i, j] = (-w[i, j][0] - w[i - 1, j][0]) / 2 + (-w[i, j][1] - w[i, j - 1][1]) / 2
            else:
                velocity_field_divergence[i, j] = (-w[i, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2
        else:
            if j == 0:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] + w[i, j][1]) / 2
            elif j == res - 1:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (-w[i, j][1] - w[i, j - 1][1]) / 2
            else:
                velocity_field_divergence[i, j] = (w[i + 1, j][0] - w[i - 1, j][0]) / 2 + (w[i, j + 1][1] - w[i, j - 1][1]) / 2

# helper function for projection
@ti.kernel
def subtract_q_gradient(velocity_field: ti.template(), q_gradient: ti.template()):
    for i, j in velocity_field:
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
        dqdx = (q_gradient[right, j] - q_gradient[left, j]) / 2
        dqdy = (q_gradient[i, up] - q_gradient[i, down]) / 2
        velocity_field[i, j] -= ti.Vector([float(dqdx), float(dqdy)])


# helper functions
@ti.kernel
def field_to_vector(field: ti.template(), vector: ti.template()):
    for i, j in field:
        vector[i * res + j] = -field[i, j]

@ti.kernel
def vector_to_field(q_vector: ti.ext_arr(), q_field: ti.template()):
    for i, j in q_field:
        q_field[i, j] = q_vector[i * res + j]


@ti.kernel
def flatten_velocity(velocity_field: ti.template(),
                     u: ti.template(),
                     v: ti.template()):
    for i, j in velocity_field:
        u[i + j * res] = velocity_field[i, j][0]
        v[i + j * res] = velocity_field[i, j][1]


@ti.kernel
def flatten_density(density_field: ti.template(),
                    r: ti.template(),
                    g: ti.template(),
                    b: ti.template()):
    for i, j in density_field:
        r[i + j * res] = density_field[i, j][0]
        g[i + j * res] = density_field[i, j][1]
        b[i + j * res] = density_field[i, j][2]


@ti.kernel
def add_to_velocity(u: ti.ext_arr(),
                    v: ti.ext_arr(),
                    velocity_field: ti.template()):
    for i, j in velocity_field:
        velocity_field[i, j][0] = u[i + j * res]
        velocity_field[i, j][1] = v[i + j * res]


@ti.kernel
def add_to_density(r: ti.ext_arr(),
                   g: ti.ext_arr(),
                   b: ti.ext_arr(),
                   density_field: ti.template()):
    for i, j in density_field:
        density_field[i, j][0] = r[i + j * res]
        density_field[i, j][1] = g[i + j * res]
        density_field[i, j][2] = b[i + j * res]


# Diffusion
# Create the place holder for sparse linear system solutions
u = ti.field(ti.f32, shape=res*res)
v = ti.field(ti.f32, shape=res*res)

r = ti.field(ti.f32, shape=res*res)
g = ti.field(ti.f32, shape=res*res)
b = ti.field(ti.f32, shape=res*res)


def diffuse_velocity(velocity_field):
    flatten_velocity(velocity_field, u, v)
    new_u = solver_velocity.solve(u)
    new_v = solver_velocity.solve(v)
    add_to_velocity(new_u, new_v, velocity_field)


def diffuse_density(density_field):
    flatten_density(density_field, r, g, b)
    new_r = solver_density.solve(r)
    new_g = solver_density.solve(g)
    new_b = solver_density.solve(b)
    add_to_density(new_r, new_g, new_b, density_field)


# projection
q = ti.field(float, shape=(res, res))
# Create the place holder for discrete Poisson equation solution
velocity_divergence_vector = ti.field(ti.f32, shape=res*res)


def solve_poisson_equation(solver, velocity_field_divergence):
    field_to_vector(velocity_field_divergence, velocity_divergence_vector)
    q_vector = solver.solve(velocity_divergence_vector)
    vector_to_field(q_vector, q)

def project(velocity_field):
    if boundary_method == 'fixed':
        divergence_fixed(velocity_field)
    elif boundary_method == 'periodic':
        divergence_periodic(velocity_field)
    else:
        print('boundary_method wrong')
    solve_poisson_equation(poisson_equation_solver, velocity_field_divergence)
    subtract_q_gradient(velocity_field, q)


# dissipation
@ti.kernel
def dissipate(density_field: ti.template()):
    for i, j in density_field:
        density_field[i, j] = density_field[i, j] / (1 + dt * dissipation_rate)


# Vstep
def Vstep():
    add_force(U0, forces)
    
    transport(U1, U0, U0)
    diffuse_velocity(U1)
    project(U1)
    dissipate(U1)


# Sstep
def Sstep():
    add_source(S0, sources)
    
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

    viscosity_rate = 0.0
    add_sources_picture(S1, img)

# =======================================GUI=====================================================
gui = ti.GUI('Stable Fluid', (res, res))
# user inputs
user_inputs = [0.0 for _ in range(7)]
while gui.running:
    # clear user inputs except color
    for i in range(len(user_inputs)):
        if i < 4:
            user_inputs[i] = 0.0
    # handle user interaction
    if gui.get_event((ti.GUI.PRESS, ti.GUI.ESCAPE)):
        gui.running = False

    if gui.is_pressed(ti.GUI.LEFT, 'r'):
        reset()

    if gui.is_pressed(ti.GUI.LMB):
        mouse_x, mouse_y = gui.get_cursor_pos()
        mouse_x *= res
        mouse_y *= res

        mouse_x = min(mouse_x, res - 1)
        mouse_y = min(mouse_y, res - 1)
        mouse_x = max(mouse_x, 0)
        mouse_y = max(mouse_y, 0)

        user_inputs[2] = mouse_x
        user_inputs[3] = mouse_y

        if cursor:
            direction = mouse_x - cursor[0], mouse_y - cursor[1]
            user_inputs[0] = direction[0]
            user_inputs[1] = direction[1]

        cursor = mouse_x, mouse_y

    else:
        cursor = None
        direction = None
        color_i = random.randint(0, len(colors) - 1)
        color = colors[color_i]
        user_inputs[4] = color[0]
        user_inputs[5] = color[1]
        user_inputs[6] = color[2]

    U0, U1 = U1, U0
    S0, S1 = S1, S0

    interactive(np.array(user_inputs, np.float32), forces, sources)

    Vstep()
    Sstep()
    gui.set_image(S0)
    gui.show()
