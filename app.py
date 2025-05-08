from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.optimize import fsolve

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    delta_P = float(data['delta_P'])
    L = float(data['L'])
    rho = float(data['rho'])
    D = float(data['D'])
    mu = float(data['mu'])
    kappa = float(data['kappa'])


    v_blasius_guess =  Blasius_guess(delta_P, D, L, rho, mu)

    f_blasius_guess = Blasius(Re(rho, v_blasius_guess, D, mu))


   

    return jsonify({'blasius_guess': v_blasius_guess,'Re': Re_Num, 'f': f})



def Re(rho, v, D, mu):
    return (rho * v * D) / mu

def Blasius(Re):
    return 0.0791 / (Re**0.25)

def Blasius_guess(delta_P, D, L, rho, mu):
    numerator = delta_P * D**(5/4)
    denominator = 0.1582 * L * rho**(3/4) * mu**(1/4)
    v = (numerator / denominator)**(4/7)
    return v

def Colebrook_recursive(f, kappa, D, v, rho, mu):

    Re = Re(rho, v, D, mu)

    # Returns the difference between the LHS and RHS (we want this to be 0) 
    return 1/np.sqrt(f) + 4.0 * np.log10((kappa/D)/3.7 + 1.256 / (Re * np.sqrt(f)))


def solve_colebrook(f_blasius_guess, kappa, D, delta_P, L, rho, mu):
    
    

    f_solution, = fsolve(Colebrook_recursive, f_blasius_guess, args=(kappa, D, Re))

    return f_solution



if __name__ == '__main__':
    app.run(debug=True)

