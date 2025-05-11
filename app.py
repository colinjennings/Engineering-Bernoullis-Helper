from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from scipy.optimize import fsolve
import os
import math

app = Flask(__name__)
CORS(app)



@app.route('/', methods=['GET'])
def home():
    return send_file('templates/index.html')


@app.route('/calculate', methods=['POST'])
def calculate_loss():

    data = request.json
    
    # loss type is string
    loss_type = data.get('lossType')

    # everything else is float
    L = float(data.get('L')  or 0)
    D = float(data.get('D')  or 0)
    rho = float(data.get('rho') or 0)
    mu = float(data.get('mu') or 0)
    kappa = float(data.get('kappa') or 0)
    k = float(data.get('k') or 0)
    g = float(data.get('g'))
    head_loss = float(data.get('head_loss'))


    # based on loss type, calculate results
    if loss_type == 'major':

        v_sol, re_sol, f_sol = solve_major_loss(head_loss, L, D, g, rho, kappa, mu)

        results = {
            'velocity': round_sig(v_sol),
            'reynoldsNumber': round_sig(re_sol),
            'frictionFactor': round_sig(f_sol)
        }
    elif loss_type == 'minor':

        v_sol = solve_minor_loss(head_loss, k, g)

        results = {
            'velocity': round_sig(v_sol)
        }
    elif loss_type == 'both':

        v_sol, re_sol, f_sol, major_loss, minor_loss = solve_both_loss(head_loss, L, D, g, rho, kappa, mu, k)

        results = {
            'velocity': round_sig(v_sol),
            'reynoldsNumber': round_sig(re_sol),
            'frictionFactor': round_sig(f_sol),
            'majorLossTerm': round_sig(major_loss),
            'minorLossTerm': round_sig(minor_loss)
        }
    else:
        return jsonify({'error': 'Invalid loss type'}), 400

    # return the results
    return jsonify(results), 200

    
def round_sig(x, sig=4):
    """
    Rounds x to sig # of sig figs, default 4
    """
    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

    
def Re(rho, v, D, mu):
    """
    Calculate the Reynolds number for flow in a pipe.

    Parameters:
    - rho: Fluid density
    - v: Fluid velocity
    - D: Pipe diameter
    - mu: Fluid dynamic viscosity

    Returns:
    - Reynolds number
    """
    return (rho * v * D) / mu


def colebrook(Re, kappa, D):
    """
    Calculate the Darcy-Weisbach friction factor using the Colebrook-White equation.
    
    Parameters:
    - Re: Reynolds number
    - kappa: Pipe roughness
    - D: Pipe diameter

    Returns:
    - Friction factor
    """
    def diff(f, kappa, D, Re):
        f = f[0]
        LHS = 1 / np.sqrt(f)
        RHS = -4.0 * np.log10((kappa / D) / 3.7 + 1.256 / (Re * np.sqrt(f)))
        return LHS - RHS

    return fsolve(diff, 0.00000001, args=(kappa, D, Re))[0]


def solve_pressure_drop(delta_P, L, rho, D, mu, kappa):
    """
    Calculate the fluid velocity, Reynolds number, and friction factor given a pressure drop.
    
    Parameters:
    - delta_P: Pressure drop
    - L: Pipe length
    - rho: Fluid density
    - D: Pipe diameter
    - mu: Fluid dynamic viscosity
    - kappa: Pipe roughness

    Returns:
    - Fluid velocity, Reynolds number, friction factor
    """
    def major_loss_diff(v):
        re = Re(rho, v, D, mu)
        f = colebrook(re, kappa, D)
        return (delta_P / L) - (2 * f * rho * v**2) / D

    v_sol = fsolve(major_loss_diff, 0.0001)[0]
    re_sol = Re(rho, v_sol, D, mu)
    f_sol = colebrook(re_sol, kappa, D)
    
    return v_sol, re_sol, f_sol


def solve_major_loss(head_loss, L, D, g, rho, kappa, mu):
    """
    Calculate the fluid velocity, Reynolds number, and friction factor given a head loss.
    
    Parameters:
    - head_loss: Head loss
    - L: Pipe length
    - D: Pipe diameter
    - g: Gravitational acceleration
    - rho: Fluid density
    - kappa: Pipe roughness
    - mu: Fluid dynamic viscosity

    Returns:
    - Fluid velocity, Reynolds number, friction factor
    """
    def diff(v):
        re = Re(rho, v, D, mu)
        f = colebrook(re, kappa, D)
        return head_loss - (2 * f * L * v**2) / (D * g)

    v_sol = fsolve(diff, 0.0001)[0]
    re_sol = Re(rho, v_sol, D, mu)
    f_sol = colebrook(re_sol, kappa, D)
    
    return v_sol, re_sol, f_sol


def solve_minor_loss(head_loss, k, g):
    """
    Calculate the fluid velocity given a minor head loss.
    
    Parameters:
    - head_loss: Minor head loss
    - k: Minor loss coefficient
    - g: Gravitational acceleration

    Returns:
    - Fluid velocity
    """

    v_sol = np.sqrt((head_loss * 2 * g) / k)
    return v_sol


def get_minor_loss(v, k, g):
    """
    Calculate the minor loss from velocity solution

    Parameters:
    - v: Fluid velocity
    - k: Minor loss coefficient
    - g: Gravitational acceleration

    Returns:
    - Minor head loss
    """
    return (k * v**2) / (2*g)


def solve_both_loss(head_loss, L, D, g, rho, kappa, mu, k):
    """
    Calculate the fluid velocity, Reynolds number, and friction factor given both major and minor losses.
    
    Parameters:
    - head_loss: Total head loss
    - L: Pipe length
    - D: Pipe diameter
    - g: Gravitational acceleration
    - rho: Fluid density
    - kappa: Pipe roughness
    - mu: Fluid dynamic viscosity
    - k: Minor loss coefficient

    Returns:
    - Fluid velocity, Reynolds number, friction factor, majojr head loss, minor head loss
    """
    def diff(v):
        re = Re(rho, v, D, mu)
        f = colebrook(re, kappa, D)
        major_loss = (2 * f * L * v**2) / (D * g)
        minor_loss = (k * v**2) / (2 * g)
        return head_loss - major_loss - minor_loss

    v_sol = fsolve(diff, 0.0001)[0]
    re_sol = Re(rho, v_sol, D, mu)
    f_sol = colebrook(re_sol, kappa, D)
    minor_loss = get_minor_loss(v_sol, k, g)
    major_loss = head_loss - minor_loss
    
    return v_sol, re_sol, f_sol, major_loss, minor_loss



if __name__ == '__main__':
    app.run(port=5000, debug=True)