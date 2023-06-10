import time
import chaospy
import warnings
import numpy as np
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split

import heat
import metrics
import oscillator

sns.set_context('notebook', font_scale=1.2)
sns.set_style('ticks')
sns.set_palette('bright')
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/content-creation/main/nma.mplstyle")
warnings.filterwarnings("ignore", category=UserWarning)


def example1(**kwargs):
    """
    Example 1.1: Damped linear oscillator model. PC expansion using the least squares method.
    """

    # Define default values
    default_values = {'train_size': 750, 'test_size': 250, 'c_lower': 0.08, 'c_upper': 0.12, 't_min': 0,
                      't_max': 30, 'coord_num': 1000, 'max_order': 2, 'plot': False, 'quantiles': False}

    # Define and initialize local variables
    for key, value in default_values.items():
        if key not in kwargs:
            kwargs[key] = default_values[key]

    # Define the damped linear oscillator model with default settings
    model = oscillator.DampedOscillator()
    model.solve()

    # Define the distribution of the damping factor
    c_dist = chaospy.Uniform(kwargs['c_lower'], kwargs['c_upper'])

    # Sample from the distribution using latin hypercube experimental design
    c_sample = c_dist.sample(kwargs['train_size'] + kwargs['test_size'], rule='latin_hypercube')

    # Define the time point sequence
    coordinates = np.linspace(kwargs['t_min'], kwargs['t_max'], kwargs['coord_num'])

    # Evaluate the model at the experimental design points
    model_eval = np.array([model.compute_displacement(p, coordinates) for p in c_sample])

    # Create train and test sets
    train_sample, test_sample, model_train_eval, model_test_eval = train_test_split(c_sample, model_eval,
                                                                                    test_size=kwargs['test_size'],
                                                                                    train_size=kwargs['train_size'])

    # Define the PC basis
    chaos_basis = chaospy.generate_expansion(kwargs['max_order'], c_dist, normed=True)

    # Define the PC metamodel
    metamodel = chaospy.fit_regression(chaos_basis, train_sample.reshape(1, -1), model_train_eval)

    # Evaluate the metamodel at the testing points
    metamodel_test_eval = metamodel(test_sample).T

    # Make comparative plots
    if kwargs['plot']:
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 13))
        ax[0].plot(coordinates, model_test_eval.T, alpha=0.01, color='red')
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('$\\mathcal{M}(t, c)$')
        ax[0].set_title('Оценки модели при случайном параметре $c$')

        ax[1].plot(coordinates, metamodel_test_eval.T, alpha=0.01, color='blue')
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('$\\mathcal{M}_{PC}(t, c)$')
        ax[1].set_title('Оценки метамодели ПХ при случайном параметре $c$')

        ax[2].plot(coordinates, model_test_eval.T, alpha=0.01, color='red')
        ax[2].plot(coordinates, metamodel_test_eval.T, alpha=0.01, color='blue')
        ax[2].set_xlabel('t')
        ax[2].set_ylabel('$y$')
        ax[2].set_title('Совмещённые оценки')
        plt.savefig('res/Oscillator evaluation.jpg', dpi=500)

    # Compute the errors
    e_loo = metrics.leave_one_out_error(test_sample, chaos_basis, model_test_eval, metamodel_test_eval)
    r_sqr = metrics.r_squared(model_test_eval, metamodel_test_eval)
    e_mae = metrics.mean_absolute_error(model_test_eval, metamodel_test_eval)
    e_rmae = metrics.relative_mean_absolute_error(model_test_eval, metamodel_test_eval)

    # Make comparative plots for the errors
    if kwargs['plot']:
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        ax[0, 0].plot(coordinates, e_loo, linewidth=2, color='forestgreen')
        ax[0, 0].set_xlabel('t')
        ax[0, 0].set_ylabel('$E_{LOO}$')
        ax[0, 0].set_title('Ошибка leave-one-out')

        ax[0, 1].plot(coordinates, r_sqr, linewidth=2, color='forestgreen')
        ax[0, 1].set_xlabel('t')
        ax[0, 1].set_ylabel('$R^2$')
        ax[0, 1].set_title('Коэффициент детерминации')
        ax[0, 1].set_yticklabels([0.8, 0.85, 0.9, 0.95, 1.0])

        ax[1, 0].plot(coordinates, e_mae, linewidth=2, color='forestgreen')
        ax[1, 0].set_xlabel('t')
        ax[1, 0].set_ylabel('$MAE$')
        ax[1, 0].set_title('Средняя абсолютная ошибка')

        ax[1, 1].plot(coordinates, e_rmae, linewidth=2, color='forestgreen')
        ax[1, 1].set_xlabel('t')
        ax[1, 1].set_ylabel('$rMAE$')
        ax[1, 1].set_title('Относительная средняя абсолютная ошибка')
        plt.savefig('res/Oscillator errors.jpg', dpi=500)

    # Compute the mean and the standard deviation of the metamodel output
    metamodel_eval_mean = chaospy.E(metamodel, c_dist)
    metamodel_eval_std = chaospy.Std(metamodel, c_dist)

    # Plot the mean and the standard deviation in a demonstrative form
    if kwargs['plot']:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.fill_between(coordinates, metamodel_eval_mean - metamodel_eval_std,
                        metamodel_eval_mean + metamodel_eval_std, alpha=0.4,
                        label='Стандартное отклонение')
        ax.plot(coordinates, metamodel_eval_mean, label='Математическое ожидание')
        ax.set_title('Математическое ожидание и стандартное отклонение')
        ax.set_xlabel('t')
        ax.set_ylabel('$\\mathcal{M}_{PC}(t, c)$')
        plt.legend()
        plt.savefig('res/Oscillator expectation.jpg', dpi=500)

    # Measure the time for computing the model and metamodel responses
    if kwargs['quantiles']:
        model_time = np.array([])
        metamodel_time = np.array([])
        for p in c_dist.sample(1000, rule='latin_hypercube'):
            start1 = time.time()
            model.compute_displacement(p, coordinates)
            finish1 = time.time()
            model_time = np.append(model_time, finish1 - start1)
            start2 = time.time()
            metamodel(p)
            finish2 = time.time()
            metamodel_time = np.append(metamodel_time, finish2 - start2)

        # Estimate the 95% quantiles
        with open('res/Oscillator quantiles.txt', 'w') as res_file:
            print(np.quantile(model_time, q=0.025).round(10),
                  np.quantile(model_time, q=0.975).round(10),
                  file=res_file)
            print(np.quantile(metamodel_time, q=0.025).round(10),
                  np.quantile(metamodel_time, q=0.975).round(10),
                  file=res_file)

    """
    Example 1.2: Damped linear oscillator model. PC expansion using the Galerkin method.
    """

    # Define the mean and the standard deviation of the damping factor
    c_mu = (kwargs['c_lower'] + kwargs['c_upper']) / 2
    c_sigma = (kwargs['c_upper'] - kwargs['c_lower']) / 2

    # Define the initial condition for the Galerkin method
    ic = oscillator.galerkin_ic(model.y_0, model.y_1, chaos_basis)

    # Compute the expansion coefficients using the Galerkin method
    galerkin_coeffs = solve_ivp(fun=oscillator.galerkin_system, t_span=(kwargs['t_min'], kwargs['t_max']),
                                y0=ic, method='RK23', t_eval=coordinates,
                                args=(c_dist, c_mu, c_sigma, chaos_basis, model))['y']

    # Define the intrusive metamodel from the result
    intrusive_metamodel = chaospy.sum(chaos_basis * galerkin_coeffs.T[:, :3], -1)

    res = {'model': model, 'chaos_basis': chaos_basis, 'metamodel': metamodel,
           'intrusive_metamodel': intrusive_metamodel, 'e_loo': e_loo, 'r_sqr': r_sqr,
           'e_mae': e_mae, 'e_rmae': e_rmae}
    return res


def example2(**kwargs):
    """
    Example 2: Heat conduction finite element model. PC expansion using the least squares method.
    """

    # Define default values
    default_values = {'n_x': 5, 'n_y': 2, 'train_size': 750, 'test_size': 250, 'q_lower': 3000, 'q_upper': 3200,
                      'Tg_mean': 20, 'Tg_sigma': 1, 'alpha_g_lower': 30, 'alpha_g_upper': 90, 'lambda_mu': 35,
                      'lambda_sigma': 1.5, 'max_order': 2, 'plot': False, 'errors': False, 'quantiles': False}

    # Define and initialize local variables
    for key, value in default_values.items():
        if key not in kwargs:
            kwargs[key] = default_values[key]

    # Define the determined parameters model
    model = heat.FiniteElementModel()
    model.discretize(kwargs['n_x'], kwargs['n_y'], plot=kwargs['plot'])
    model.compute_temperature(plot=kwargs['plot'], mesh=True)

    # Plot the temperature distribution
    if kwargs['plot']:
        x = np.linspace(0, model.width, kwargs['n_x'] + 1)
        y = np.linspace(0, model.height, kwargs['n_y'] + 1)
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.pcolormesh(X, Y, model.temp_mesh, shading='gouraud', cmap='plasma')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.title('Распределение температур в теле', y=1.1)
        plt.savefig('res/Heat temperature (determined).jpg', dpi=500)

    # Define the distribution of the stochastic factors
    q_dist = chaospy.Uniform(kwargs['q_lower'], kwargs['q_upper'])
    Tg_dist = chaospy.Normal(kwargs['Tg_mean'], kwargs['Tg_sigma'])
    alpha_g_dist = chaospy.Uniform(kwargs['alpha_g_lower'], kwargs['alpha_g_upper'])
    lambda_dist = chaospy.Normal(kwargs['lambda_mu'], kwargs['lambda_sigma'])
    joint_dist = chaospy.J(q_dist, Tg_dist, alpha_g_dist, lambda_dist)

    # Sample from the joint distribution using sobol experimental design
    joint_sample = joint_dist.sample(kwargs['train_size'] + kwargs['test_size'], rule='sobol')

    # Plot the experimental design
    if kwargs['plot']:
        fig, ax = plt.subplots(2, 2, figsize=(20, 20), subplot_kw=dict(projection='3d'))
        axes = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        colors = ['red', 'dodgerblue', 'forestgreen', 'purple']
        labels = [r'q', r'$\mathbf{T_g}$', r'$\mathbf{\alpha_g}$', r'$\mathbf{\lambda}$']
        for i in range(4):
            ax[i // 2, i % 2].scatter(*joint_sample[axes[i]], marker='o', s=15, color=colors[i])
            ax[i // 2, i % 2].view_init(elev=38, azim=39)
            ax[i // 2, i % 2].set_xlabel(labels[axes[i][0]], fontsize=30, weight='bold', labelpad=15)
            ax[i // 2, i % 2].set_ylabel(labels[axes[i][1]], fontsize=30, weight='bold', labelpad=15)
            ax[i // 2, i % 2].zaxis.set_rotate_label(False)
            ax[i // 2, i % 2].set_zlabel(labels[axes[i][2]], rotation=0, fontsize=30, weight='bold', labelpad=7)
            ax[i // 2, i % 2].tick_params(labelsize=20)

        fig.suptitle('Экспериментальный дизайн', fontsize=40)
        plt.savefig('res/Heat experimental design.jpg', dpi=500)

    # Evaluate the model at the experimental design points
    model_eval = []
    model_param = {'width': 0.4, 'height': 0.2, 'thickness': 0.1}
    model = heat.FiniteElementModel(**model_param)
    model.discretize(kwargs['n_x'], kwargs['n_y'])
    for p in tqdm(joint_sample.T):
        model.q = p[0]
        model.Tg = p[1]
        model.alpha_g = p[2]
        model.lambda_x = model.lambda_y = p[3]
        model.compute_temperature()
        model_eval.append(np.reshape(model.temp, -1))

    # Create train and test sets
    train_sample, test_sample, model_train_eval, model_test_eval = train_test_split(joint_sample.T, model_eval,
                                                                                    test_size=kwargs['test_size'],
                                                                                    train_size=kwargs['train_size'])

    # Normalize the inputs
    train_sample_mean = np.mean(train_sample, axis=0)
    train_sample_std = np.std(train_sample, axis=0)
    train_sample_normalized = (train_sample - train_sample_mean) / train_sample_std
    test_sample_normalized = (test_sample - train_sample_mean) / train_sample_std

    # Define the PC basis
    chaos_basis = chaospy.generate_expansion(kwargs['max_order'], joint_dist, normed=True)

    # Define the PC metamodel
    metamodel = chaospy.fit_regression(chaos_basis, train_sample_normalized.T, model_train_eval)

    # Evaluate the metamodel at the testing points
    metamodel_test_eval = metamodel(*test_sample_normalized.T).T

    # Compute the errors
    e_emp = np.mean(metrics.empirical_error(model_test_eval, metamodel_test_eval))
    r_sqr = np.mean(metrics.r_squared(model_test_eval, metamodel_test_eval))
    e_mae = np.mean(metrics.mean_absolute_error(model_test_eval, metamodel_test_eval))
    e_rmae = np.mean(metrics.relative_mean_absolute_error(model_test_eval, metamodel_test_eval))

    if kwargs['errors']:
        with open('res/Heat errors.txt', 'w') as res_file:
            print(e_emp, file=res_file)
            print(r_sqr, file=res_file)
            print(e_mae, file=res_file)
            print(e_rmae, file=res_file)

    # Measure the time for computing the model and metamodel responses
    if kwargs['quantiles']:
        model_time = np.array([])
        metamodel_time = np.array([])
        model_param = {'width': 0.4, 'height': 0.2, 'thickness': 0.1}
        model = heat.FiniteElementModel(**model_param)
        model.discretize(kwargs['n_x'], kwargs['n_y'])
        for p in joint_dist.sample(100, rule='sobol').T:
            model.q = p[0]
            model.Tg = p[1]
            model.alpha_g = p[2]
            model.lambda_x = model.lambda_y = p[3]
            start1 = time.time()
            model.compute_temperature()
            finish1 = time.time()
            model_time = np.append(model_time, finish1 - start1)
            start2 = time.time()
            metamodel(*p)
            finish2 = time.time()
            metamodel_time = np.append(metamodel_time, finish2 - start2)

        # Estimate the 95% quantiles
        with open('res/Heat quantiles.txt', 'w') as res_file:
            print(np.quantile(model_time, q=0.025).round(10),
                  np.quantile(model_time, q=0.975).round(10),
                  file=res_file)
            print(np.quantile(metamodel_time, q=0.025).round(10),
                  np.quantile(metamodel_time, q=0.975).round(10),
                  file=res_file)

    # Make the complexity plot
    if kwargs['plot']:
        model_time = np.array([])
        metamodel_time = np.array([])
        points = np.linspace(1, kwargs['test_size'], 10, dtype=int)
        for p in points:
            start1 = time.time()
            model_param = {'width': 0.4, 'height': 0.2, 'thickness': 0.1}
            model = heat.FiniteElementModel(**model_param)
            model.discretize(kwargs['n_x'], kwargs['n_y'])
            for s in tqdm(joint_dist.sample(p, rule='sobol').T):
                model.q = s[0]
                model.Tg = s[1]
                model.alpha_g = s[2]
                model.lambda_x = model.lambda_y = s[3]
                model.compute_temperature()
            finish1 = time.time()
            start2 = time.time()
            for s in joint_dist.sample(p, rule='sobol').T:
                metamodel(*s)
            finish2 = time.time()
            model_time = np.append(model_time, finish1 - start1)
            metamodel_time = np.append(metamodel_time, finish2 - start2)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(points, model_time, color='red', linewidth=2, label='Модель')
        ax.plot(points, metamodel_time, color='blue', linewidth=2, label='Метамодель')
        fig.suptitle('Сложности модели и метамодели')
        ax.set_xlabel('N')
        ax.set_ylabel('Время выполнения, с')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.legend()
        plt.savefig('res/Heat time complexity.jpg', dpi=500)

    res = {'model': model, 'chaos_basis': chaos_basis, 'metamodel': metamodel,
           'e_emp': e_emp, 'r_sqr': r_sqr, 'e_mae': e_mae, 'e_rmae': e_rmae}
    return res


if __name__ == '__main__':
    """
    Example 1: Damped linear oscillator model.
    """

    res1 = example1()
    
    """
    Example 2: Heat conduction finite element model. PC expansion using the least squares method.
    """

    res2 = example2()
