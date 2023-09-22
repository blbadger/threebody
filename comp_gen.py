import typing
import pyperclip

def generate_string(a: str, b: str, c: str, d: str, t: str, , m: str, prime: bool) -> str:
    if prime:
        e = '_prime'
    else:
        e = ''
    first_denom =  f'sqrt(({a}{e}_x[i] - {b}{e}_x[i])*({a}{e}_x[i] - {b}{e}_x[i]) + ({a}{e}_y[i] - {b}{e}_y[i])*({a}{e}_y[i] - {b}{e}_y[i]) + ({a}{e}_z[i] - {b}{e}_z[i])*({a}{e}_z[i] - {b}{e}_z[i]))' 
    second_denom = f'sqrt(({c}{e}_x[i] - {d}{e}_x[i])*({c}{e}_x[i] - {d}{e}_x[i]) + ({c}{e}_y[i] - {d}{e}_y[i])*({c}{e}_y[i] - {d}{e}_y[i]) + ({c}{e}_z[i] - {d}{e}_z[i])*({c}{e}_z[i] - {d}{e}_z[i]))'

    template = f'''-9.8 * m_{m} * ({a}{e}_{t}[i] - {b}{e}_{t}[i]) / ({first_denom}*{first_denom}*{first_denom}) -9.8 * m_{m} * ({c}{e}_{t}[i] - {d}{e}_{t}[i]) / ({second_denom}*{second_denom}*{second_denom});'''

    pyperclip.copy(template)
    return template


a = 'p3'
b = 'p1'
c = 'p3'
d = 'p2'
m = '3'
t = 'z'
print (generate_string(a, b, c, d, t, prime=True))

