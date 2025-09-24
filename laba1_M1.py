import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 10  # м/с^2



def derivatives_linear(t, state, c):
    x, y, vx, vy = state
    ax = -c * vx
    ay = -c * vy - g
    return [vx, vy, ax, ay]

def derivatives_quadratic(t, state, c):
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    ax = -c * v * vx
    ay = -c * v * vy - g
    return [vx, vy, ax, ay]




def simulate_projectile(v0, theta_deg, model='none', c=0.0, t_max=200.0, n_points=1000):
    y0 = 0.0
    theta = np.deg2rad(theta_deg)
    v0x, v0y = v0*np.cos(theta), v0*np.sin(theta)
    state0 = [0.0, y0, v0x, v0y]

    def hit_ground(t, y):
        return y[1]
    hit_ground.direction = -1
    hit_ground.terminal = True

    if model == 'linear':
        deriv = lambda t, y: derivatives_linear(t, y, c)
    elif model == 'quadratic':
        deriv = lambda t, y: derivatives_quadratic(t, y, c)
    else:
        # без сопротивления
        deriv = lambda t, y: [y[2], y[3], 0, -g]

    sol = solve_ivp(deriv,
                    t_span=(0, t_max), y0=state0, events=hit_ground,
                    t_eval=np.linspace(0, t_max, n_points))

    if sol.t_events[0].size > 0:
        t_impact = float(sol.t_events[0][0])
        x_impact = float(sol.y_events[0][0][0])
        mask = sol.t <= t_impact
        t = sol.t[mask]
        x = sol.y[0, mask]
        y = sol.y[1, mask]
        t = np.append(t, t_impact)
        x = np.append(x, x_impact)
        y = np.append(y, 0.0)
    else:
        x_impact = None
        t = sol.t
        x = sol.y[0]
        y = sol.y[1]

    return {'t': t, 'x': x, 'y': y, 'x_impact': x_impact}




#теор
def theoretical_range(v0, theta_deg, g=10):
    theta = np.deg2rad(theta_deg)
    return v0**2 * np.sin(2*theta) / g



#ввод
v0 = float(input("Введите начальную скорость v0 (м/с): "))
theta = float(input("Введите угол броска θ (градусы): "))
c_linear = float(input("Введите коэффициент линейного сопротивления воздуха c1: "))
c_quad = float(input("Введите коэффициент квадратичного сопротивления воздуха c2: "))



#cсимуляция
res_none = simulate_projectile(v0, theta, model='none')
res_linear = simulate_projectile(v0, theta, model='linear', c=c_linear)
res_quad = simulate_projectile(v0, theta, model='quadratic', c=c_quad)



#график
plt.figure(figsize=(10,6))
plt.plot(res_none['x'], res_none['y'], label=f"Без сопротивления (R={res_none['x_impact']:.2f} м)")
plt.plot(res_linear['x'], res_linear['y'], label=f"Линейное F~v (R={res_linear['x_impact']:.2f} м)")
plt.plot(res_quad['x'], res_quad['y'], label=f"Квадратичное F~v² (R={res_quad['x_impact']:.2f} м)")
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel("x, м")
plt.ylabel("y, м")
plt.title("Сравнение траекторий с разными моделями сопротивления воздуха")
plt.legend()
plt.grid(True)
plt.show()


#вывод
print(' ')
print('Результаты:')
print(f"Теоретическая дальность без сопротивления: {theoretical_range(v0, theta):.2f} м")
print(f"Реальная дальность без сопротивления: {res_none['x_impact']:.2f} м")
print(f"Реальная дальность с линейным сопротивлением: {res_linear['x_impact']:.2f} м")
print(f"Реальная дальность с квадратичным сопротивлением: {res_quad['x_impact']:.2f} м")
