import io
import sys
import sympy as sp
import numpy as np
from typing import Union
import matplotlib.pyplot as plt


class FiniteElementModel:
    acceptable_keys_list = ['alpha_g', 'Tg', 'q', 'width', 'height', 'thickness', 'lambda_x', 'lambda_y']

    def __init__(self, verbose: bool = False, **kwargs):
        """
        Creates the finite element model for the heat conduction task.

        :param kwargs:    dict, of the model parameters
                     :alpha_g:     float [W/(sm^2 * °C)],  heat transfer coefficient
                     :Tg:          float [°C],             temperature of the surrounding environment
                     :q:           float [W/sm^2],         heat flux
                     :width:       float [sm],             width of the body
                     :height:      float [sm],             height of the body
                     :thickness:   float [sm],             thickness of the body
                     :lambda_x:    float [W/(sm * °C)],    thermal conductivity of the material in
                                                           the x-axis direction
                     :lambda_y:    float [W/(sm * °C)],    thermal conductivity of the material in
                                                           the y-axis direction
                     :verbose:     bool,                   flag to print in the console a detailed report
        """

        if len(kwargs) == 0:
            kwargs = {'alpha_g': 60, 'Tg': 20, 'q': 3100, 'width': 0.4, 'height': 0.2,
                      'thickness': 0.1, 'lambda_x': 35, 'lambda_y': 35}
        else:
            if not all(map(lambda x: isinstance(x, (int, float)), kwargs.values())):
                raise TypeError("Parameter values of the model must be of the int or float type.")

            if any(map(lambda x: np.isnan(x) or np.isinf(x), kwargs.values())):
                raise ValueError("All parameters of the system should be well-defined.")

        self.acceptable_keys_list_copy = self.acceptable_keys_list.copy()
        for k in kwargs.keys():
            if k in self.acceptable_keys_list_copy:
                self.__setattr__(k, kwargs[k])

        if len(self.acceptable_keys_list_copy) != 0:
            for k in self.acceptable_keys_list_copy:
                self.__setattr__(k, None)

        # Future attributes
        self.n_x = None
        self.n_y = None
        self.s_e = None
        self.upper_triangles = None
        self.upper_triangles_idx = None
        self.lower_triangles = None
        self.lower_triangles_idx = None
        self.k_glob = None
        self.f_glob = None
        self.temp = None
        self.temp_mesh = None
        self.n_array_up = None
        self.n_array_low = None
        self.verbose = verbose

    def __setattr__(self, key: str, value: Union[int, float]):
        """
        Defines the attribute setter.

        :param key:    str,                     field to be assigned
        :param value:  Union[int, float],       value to be assigned
        """

        positive_keys_list = ['width', 'height', 'thickness', 'lambda_x', 'lambda_y']
        if (value is not None) and ((key in positive_keys_list) and not (value > 0)):  # to catch NaN
            raise ValueError(f"{key} parameter of the system should be positive.")

        if key in self.acceptable_keys_list:
            if (value is not None) and (self.__dict__.get(f'_{key}') is None):
                self.acceptable_keys_list_copy.remove(key)
            self.__dict__[f'_{key}'] = value
        else:
            self.__dict__[f'{key}'] = value

    def __getattr__(self, item: str) -> float:
        """
        Defines the attribute getter.

        :param item:  str,     name of the field to be returned
        :return:      float,   value of the field
        """
        if item in self.acceptable_keys_list:
            return self.__dict__[f'_{item}']
        else:
            return self.__dict__[f'{item}']

    def discretize(self, n_x: int, n_y: int, plot: bool = False, loc: bool = False) -> Union[None, dict]:
        """
        Computes the discretization of the body.

        :param n_x:   int,                     number of segments on the x-axis
        :param n_y:   int,                     number of segments on the y-axis
        :param plot:  bool,                    flag to make a finite element discretization plot
        :param loc:   bool,                    flag to return the triangles coordinates and the shape functions
        :return:      Union[None, dict],       None or dictionary that contains the finite element vertices data
        """

        if not (isinstance(n_x, int)) or not (isinstance(n_y, int)):
            raise TypeError("The number of segments must be of the int type.")

        if (n_x <= 0) or (n_y <= 0):
            raise ValueError("The number of segments should be positive.")

        if (self.width is None) or (self.height is None):
            raise ValueError("The width and the height of the object must be already defined.")

        # Assign sizes to the fields
        self.n_x = n_x
        self.n_y = n_y

        # Coordinates for the element vertices
        vertices_x = np.linspace(0, self.width, n_x + 1)
        vertices_y = np.linspace(self.height, 0, n_y + 1)

        # Composing triangles
        upper_triangles = []
        upper_triangles_idx = []
        for j in range(len(vertices_y) - 1):
            for i in range(len(vertices_x) - 1):
                idx = j + (n_y + 1) * i
                first = (vertices_x[i], vertices_y[j])
                second = (vertices_x[i + 1], vertices_y[j + 1])
                third = (vertices_x[i + 1], vertices_y[j])
                upper_triangles.append([first, second, third])
                upper_triangles_idx.append([idx, idx + (n_y + 2), idx + (n_y + 1)])

        lower_triangles = []
        lower_triangles_idx = []
        for j in range(len(vertices_y) - 1):
            for i in range(len(vertices_x) - 1):
                idx = j + (n_y + 1) * i
                first = (vertices_x[i], vertices_y[j])
                second = (vertices_x[i], vertices_y[j + 1])
                third = (vertices_x[i + 1], vertices_y[j + 1])
                lower_triangles.append([first, second, third])
                lower_triangles_idx.append([idx, idx + 1, idx + (n_y + 2)])

        # Plot the discretization result
        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            fontdict = dict(color='white', fontweight='bold', fontsize='x-large', ha='center')

            for i in range(n_x * n_y):
                x_coord_up = [elem[0] for elem in upper_triangles[i]]
                y_coord_up = [elem[1] for elem in upper_triangles[i]]
                x_coord_low = [elem[0] for elem in lower_triangles[i]]
                y_coord_low = [elem[1] for elem in lower_triangles[i]]
                ax.fill(x_coord_up, y_coord_up, color='blue')
                ax.text(np.sum(x_coord_up) / 3, np.sum(y_coord_up) / 3, f'{2 * i + 2}', fontdict)
                ax.fill(x_coord_low, y_coord_low, color='red')
                ax.text(np.sum(x_coord_low) / 3, np.sum(y_coord_low) / 3, f'{2 * i + 1}', fontdict)
                for j in range(3):
                    ax.text(x_coord_up[j], y_coord_up[j], f'{upper_triangles_idx[i][j]}')
                    ax.text(x_coord_low[j], y_coord_low[j], f'{lower_triangles_idx[i][j]}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            plt.savefig('res/Heat FEM discretization.jpg', dpi=500)

        # Assign finite element vertices to the fields
        self.upper_triangles = upper_triangles
        self.upper_triangles_idx = upper_triangles_idx
        self.lower_triangles = lower_triangles
        self.lower_triangles_idx = lower_triangles_idx

        # Debugging output
        if self.verbose:
            print("\nGlobal coordinates of the upper triangles:")
            print(np.array(upper_triangles))
            print("\nGlobal indices of the upper triangles:")
            print(np.array(upper_triangles_idx))
            print("\nGlobal coordinates of the lower triangles:")
            print(np.array(lower_triangles))
            print("\nGlobal indices of the lower triangles:")
            print(np.array(lower_triangles_idx))

        # Return the result if needed
        if loc:
            res = {'upper_triangles': upper_triangles, 'lower_triangles': lower_triangles}
            return res

    def compute_temperature(self, plot: bool = False, matr: bool = False, mesh: bool = False,
                            loc: bool = False) -> dict:
        """
        Computes the temperature.

        :param plot:  bool,       flag to make a nodal temperature plot
        :param matr:  bool,       flag to return the stiffness matrix and the right hand side vector
        :param mesh:  bool,       flag to return the mesh representation of the temperature
        :param loc:   bool,       flag to return the triangles coordinates and the shape functions
        :return:      dict,       dictionary that contains the temperature and accompanying required data
        """

        if (self.n_x is None) or (self.n_y is None):
            raise RuntimeError("Discretize first to compute the nodal temperature.")

        if len(self.acceptable_keys_list_copy) != 0:
            raise ValueError(f"Set the values of lacking attributes:{self.acceptable_keys_list_copy}.")

        # Square of one element
        s_e = 1 / 2 * (self.width / self.n_x) * (self.height / self.n_y)

        # Arrays of the stiffness matrices and shape functions
        # for the upper and lower triangles respectively
        k_array_up = []
        f_array_up = []
        n_array_up = []

        k_array_low = []
        f_array_low = []
        n_array_low = []
        for i in range(self.n_x * self.n_y):
            x_coord_up = [elem[0] for elem in self.upper_triangles[i]]
            y_coord_up = [elem[1] for elem in self.upper_triangles[i]]

            if self.verbose:
                print(f'\nElement No. {2 * i + 2}:')
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            upper_element = FiniteElement(x_coord_up, y_coord_up, s_e, self)
            k_array_up.append(upper_element.stiffness_matrix)
            f_array_up.append(upper_element.right_vector)
            n_array_up.append(upper_element.shape_func)
            sys.stdout = old_stdout
            if self.verbose and len(buffer.getvalue()) == 0:
                print("Isolated finite element.")
            else:
                print(buffer.getvalue(), end='')

            x_coord_low = [elem[0] for elem in self.lower_triangles[i]]
            y_coord_low = [elem[1] for elem in self.lower_triangles[i]]

            if self.verbose:
                print(f'\nElement No. {2 * i + 1}:')
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            lower_element = FiniteElement(x_coord_low, y_coord_low, s_e, self)
            k_array_low.append(lower_element.stiffness_matrix)
            f_array_low.append(lower_element.right_vector)
            n_array_low.append(lower_element.shape_func)
            sys.stdout = old_stdout
            if self.verbose and len(buffer.getvalue()) == 0:
                print("Isolated finite element.")
            else:
                print(buffer.getvalue(), end='')

        k_array_up = np.array(k_array_up)
        f_array_up = np.array(f_array_up)
        n_array_up = np.array(n_array_up)

        k_array_low = np.array(k_array_low)
        f_array_low = np.array(f_array_low)
        n_array_low = np.array(n_array_low)

        # Global stiffness matrix and right hand side vector
        k_glob = np.zeros(((self.n_x + 1) * (self.n_y + 1), (self.n_x + 1) * (self.n_y + 1)))
        f_glob = np.zeros((self.n_x + 1) * (self.n_y + 1)).reshape(-1, 1)

        for i in range(self.n_x * self.n_y):
            idx_cur_up = self.upper_triangles_idx[i]
            for j in range(3):
                f_glob[idx_cur_up[j]] += f_array_up[i, j]
                for k in range(3):
                    k_glob[idx_cur_up[j], idx_cur_up[k]] += k_array_up[i, j, k]

            idx_cur_low = self.lower_triangles_idx[i]
            for j in range(3):
                f_glob[idx_cur_low[j]] += f_array_low[i, j]
                for k in range(3):
                    k_glob[idx_cur_low[j], idx_cur_low[k]] += k_array_low[i, j, k]

        # Find unknown temperature
        temp = np.linalg.solve(k_glob, f_glob)

        # Represent the temperature as a mesh on the surface
        temp_mesh = np.zeros((self.n_y + 1, self.n_x + 1))
        for i in range(self.n_x + 1):
            for j in range(self.n_y + 1):
                temp_mesh[j, i] = temp[j + (self.n_y + 1) * i]

        # Plot the finite element representation
        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            fontdict = dict(color='black', backgroundcolor='white', fontsize='medium', ha='center')

            # Coordinates for the element vertices
            vertices_x = np.linspace(0, self._width, self.n_x + 1)
            vertices_y = np.linspace(self._height, 0, self.n_y + 1)

            for i in range(self.n_x * self.n_y):
                x_vert, y_vert = np.meshgrid(vertices_x, vertices_y)
                x_hor = vertices_x.copy()
                y_hor = vertices_y * np.ones_like(vertices_x).reshape(-1, 1)
                ax.plot(x_vert, y_vert, color='blue', linewidth=3)
                ax.plot(x_hor, y_hor, color='blue', linewidth=3)
                for j in range(self.n_y + 1):
                    for k in range(self.n_x + 1):
                        ax.text(x_vert[j, k], y_vert[j, k], f'{temp_mesh[j, k].round(2)}', fontdict)
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            plt.savefig('res/Heat nodal temperature.jpg', dpi=500)

        # Set corresponding attributes
        self.s_e = s_e
        self.k_glob = k_glob
        self.f_glob = f_glob
        self.temp = temp
        self.temp_mesh = temp_mesh
        self.n_array_up = n_array_up
        self.n_array_low = n_array_low

        # Debugging output
        if self.verbose:
            print("\nGlobal stiffness matrix:")
            print(k_glob)
            print("\nGlobal right hand side vector:")
            print(f_glob)

        # The result dictionary
        res = {}
        if matr:
            res['k_glob'] = k_glob
            res['f_glob'] = f_glob
        if mesh:
            res['temp_mesh'] = temp_mesh
        else:
            res['temp'] = temp
        if loc:
            res['n_array_up'] = n_array_up
            res['n_array_low'] = n_array_low

        return res

    def interpolate_temperature(self, x: Union[int, float], y: Union[int, float]) -> float:
        """
        Computes the temperature at the body point by interpolation with the shape functions.

        :param x:  Union[int, float],       x-coordinate of the point
        :param y:  Union[int, float],       y-coordinate of the point
        :return:   float,                   temperature at the point
        """

        if any(map(lambda val: val is None, self.__dict__.values())):
            raise RuntimeError("You need to discretize the object and to compute the nodal temperature first.")

        if not (isinstance(x, (int, float))) or not (isinstance(y, (int, float))):
            raise TypeError("The coordinates must be of the int or float type.")

        if np.isnan([x, y]).any() or np.isinf([x, y]).any():
            raise ValueError("The coordinates must be well-defined.")

        if (x < 0) or (x > self._width) or (y < 0) or (y > self._height):
            raise ValueError("The point is out of the coordinate range.")

        temp_upper_triangles = []
        for i in range(self.n_y):
            for j in range(self.n_x):
                first = self.temp_mesh[i, j]
                second = self.temp_mesh[i + 1, j + 1]
                third = self.temp_mesh[i, j + 1]
                temp_upper_triangles.append([first, second, third])

        for i in range(self.n_x * self.n_y):
            AB = np.array(self.upper_triangles[i][1]) - np.array(self.upper_triangles[i][0])
            BC = np.array(self.upper_triangles[i][2]) - np.array(self.upper_triangles[i][1])
            CA = np.array(self.upper_triangles[i][0]) - np.array(self.upper_triangles[i][2])
            AO = np.array([x, y]) - np.array(self.upper_triangles[i][0])
            BO = np.array([x, y]) - np.array(self.upper_triangles[i][1])
            CO = np.array([x, y]) - np.array(self.upper_triangles[i][2])
            ABxAO = np.cross(AB, AO)
            BCxBO = np.cross(BC, BO)
            CAxCO = np.cross(CA, CO)
            cross_products = np.array([ABxAO, BCxBO, CAxCO])
            if (np.sum(cross_products > 0) == 3) or (np.sum(cross_products < 0) == 3) \
                    or ((np.sum(cross_products > 0) == 2) and (np.sum(cross_products == 0) == 1)) \
                    or ((np.sum(cross_products < 0) == 2) and (np.sum(cross_products == 0) == 1)) \
                    or ((np.sum(cross_products > 0) == 1) and (np.sum(cross_products == 0) == 2)) \
                    or ((np.sum(cross_products < 0) == 1) and (np.sum(cross_products == 0) == 2)):

                temp_interp_symb = np.dot(temp_upper_triangles[i], self.n_array_up[i])
                temp_interp = temp_interp_symb.subs([(sp.Symbol('x'), x), (sp.Symbol('y'), y)])

                if self.verbose:
                    print(f"\nThe point is located in the {2 * i + 2} (upper) element.")
                    print("\nAuxiliary cross products:")
                    print(ABxAO, BCxBO, CAxCO)

                return float(temp_interp)

        temp_lower_triangles = []
        for i in range(self.n_y):
            for j in range(self.n_x):
                first = self.temp_mesh[i, j]
                second = self.temp_mesh[i + 1, j]
                third = self.temp_mesh[i + 1, j + 1]
                temp_lower_triangles.append([first, second, third])

        for i in range(self.n_x * self.n_y):
            AB = np.array(self.lower_triangles[i][1]) - np.array(self.lower_triangles[i][0])
            BC = np.array(self.lower_triangles[i][2]) - np.array(self.lower_triangles[i][1])
            CA = np.array(self.lower_triangles[i][0]) - np.array(self.lower_triangles[i][2])
            AO = np.array([x, y]) - np.array(self.lower_triangles[i][0])
            BO = np.array([x, y]) - np.array(self.lower_triangles[i][1])
            CO = np.array([x, y]) - np.array(self.lower_triangles[i][2])
            ABxAO = np.cross(AB, AO)
            BCxBO = np.cross(BC, BO)
            CAxCO = np.cross(CA, CO)
            cross_products = np.array([ABxAO, BCxBO, CAxCO])
            if (np.sum(cross_products > 0) == 3) or (np.sum(cross_products < 0) == 3) \
                    or ((np.sum(cross_products > 0) == 2) and (np.sum(cross_products == 0) == 1)) \
                    or ((np.sum(cross_products < 0) == 2) and (np.sum(cross_products == 0) == 1)) \
                    or ((np.sum(cross_products > 0) == 1) and (np.sum(cross_products == 0) == 2)) \
                    or ((np.sum(cross_products < 0) == 1) and (np.sum(cross_products == 0) == 2)):

                temp_interp_symb = np.dot(temp_lower_triangles[i], self.n_array_low[i])
                temp_interp = temp_interp_symb.subs([(sp.Symbol('x'), x), (sp.Symbol('y'), y)])

                if self.verbose:
                    print(f"\nThe point is located in the {2 * i + 1} (lower) element.")
                    print("\nAuxiliary cross products:")
                    print(ABxAO, BCxBO, CAxCO)

                return float(temp_interp)


class FiniteElement:
    def __init__(self, x: Union[list, np.ndarray], y: Union[list, np.ndarray],
                 s_e: float, model: FiniteElementModel):
        """
        Creates the finite element for the heat conduction task.

        :param x:      Union[list, np.ndarray],       x-coordinates of the element vertices
        :param y:      Union[list, np.ndarray],       y-coordinates of the element vertices
        :param s_e:    float,                         square of the element
        :param model:  FiniteElementModel,            general finite element model
        """

        if not (isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray))):
            raise TypeError("The coordinates of the element nodes should be stored as a list or np.ndarray.")

        if len(x) != len(y):
            raise ValueError("The length of the coordinates should match the number of element nodes.")

        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("The x coordinates of the element nodes must be well defined.")

        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("The y coordinates of the element nodes must be well defined.")

        if not (s_e > 0):  # to catch NaN
            raise ValueError("The element square must be greater than zero.")

        if not isinstance(model, FiniteElementModel):
            raise TypeError("Teh finite element model should be provided.")

        # Element characteristics
        self.x = x
        self.y = y
        self.s_e = s_e

        # Model characteristics
        self.alpha_g = model.alpha_g
        self.Tg = model.Tg
        self.q = model.q
        self.width = model.width
        self.height = model.height
        self.thickness = model.thickness
        self.lambda_x = model.lambda_x
        self.lambda_y = model.lambda_y

        self.verbose = model.verbose

    @property
    def coef_a(self) -> np.ndarray:
        """
        Computes the vector of 'a' coefficients.
        :return:  np.ndarray,  row vector a
        """

        a = np.zeros_like(self.x)
        a[0] = self.x[1] * self.y[2] - self.x[2] * self.y[1]
        a[1] = self.x[2] * self.y[0] - self.x[0] * self.y[2]
        a[2] = self.x[0] * self.y[1] - self.x[1] * self.y[0]
        return a

    @property
    def coef_b(self) -> np.ndarray:
        """
        Computes the vector of 'b' coefficients.

        :return:  np.ndarray,  row vector b
        """

        b = np.zeros_like(self.x)
        b[0] = self.y[1] - self.y[2]
        b[1] = self.y[2] - self.y[0]
        b[2] = self.y[0] - self.y[1]
        return b

    @property
    def coef_c(self) -> np.ndarray:
        """
        Computes the vector of 'c' coefficients.

        :return:  np.ndarray,  row vector c
        """

        c = np.zeros_like(self.x)
        c[0] = self.x[2] - self.x[1]
        c[1] = self.x[0] - self.x[2]
        c[2] = self.x[1] - self.x[0]
        return c

    @property
    def shape_func(self) -> np.ndarray:
        """
        Computes the vector of shape functions for the element.

        :return:  np.ndarray,  shape functions N(e)
        """

        x = sp.Symbol('x')
        y = sp.Symbol('y')
        return 1 / (2 * self.s_e) * (self.coef_a + self.coef_b * x + self.coef_c * y)

    @property
    def derivative_matrix(self) -> np.ndarray:
        """
        Computes the matrix with the shape functions derivatives for the element.

        :return:  np.ndarray,  derivatives of the shape functions N(e)
        """

        return 1 / (2 * self.s_e) * np.array([self.coef_b, self.coef_c])

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """
        Computes the stiffness matrix for the element based on its location.

        :return:  np.ndarray,  stiffness matrix K(e)
        """

        k_mat = self.lambda_x * self.thickness / (4 * self.s_e) * np.outer(self.coef_b, self.coef_b) + \
                self.lambda_y * self.thickness / (4 * self.s_e) * np.outer(self.coef_c, self.coef_c)

        if np.sum(np.isclose(self.y, self.height)) == 2:
            shape_func_subs = np.array([elem.subs(sp.Symbol('y'), self.height) for elem in self.shape_func])
            int_expr_1 = self.alpha_g * self.thickness * np.outer(shape_func_subs, shape_func_subs)
            integral_1 = np.zeros_like(k_mat)
            for i in range(integral_1.shape[0]):
                for j in range(integral_1.shape[1]):
                    integral_1[i, j] = sp.integrate(int_expr_1[i, j],
                                                    (sp.Symbol('x'), np.min(self.x), np.max(self.x)))
            k_mat += integral_1

            if self.verbose:
                print("Interaction with the environment on the top.")

        if np.sum(np.isclose(self.x, self.width)) == 2:
            shape_func_subs = np.array([elem.subs(sp.Symbol('x'), self.width) for elem in self.shape_func])
            int_expr_2 = self.alpha_g * self.thickness * np.outer(shape_func_subs, shape_func_subs)
            integral_2 = np.zeros_like(k_mat)
            for i in range(integral_2.shape[0]):
                for j in range(integral_2.shape[1]):
                    integral_2[i, j] = sp.integrate(int_expr_2[i, j],
                                                    (sp.Symbol('y'), np.min(self.y), np.max(self.y)))
            k_mat += integral_2

            if self.verbose:
                print("Interaction with the environment from the right.")

        return k_mat

    @property
    def right_vector(self) -> np.ndarray:
        """
        Computes the right hand side vector for the element based on its location.

        :return:  np.ndarray,  column vector f(e)
        """

        f_vec = np.zeros_like(self.x).reshape(-1, 1)

        if np.sum(np.isclose(self.y, self.height)) == 2:
            shape_func_subs = np.array([elem.subs(sp.Symbol('y'), self.height) for elem in self.shape_func])
            int_expr_1 = self.thickness * self.alpha_g * self.Tg * shape_func_subs
            integral_1 = np.zeros_like(f_vec)
            for i in range(integral_1.shape[0]):
                integral_1[i, 0] = sp.integrate(int_expr_1[i],
                                                (sp.Symbol('x'), np.min(self.x), np.max(self.x)))
            f_vec += integral_1

        if np.sum(np.isclose(self.x, self.width)) == 2:
            shape_func_subs = np.array([elem.subs(sp.Symbol('x'), self.width) for elem in self.shape_func])
            int_expr_2 = self.thickness * self.alpha_g * self.Tg * shape_func_subs
            integral_2 = np.zeros_like(f_vec)
            for i in range(integral_2.shape[0]):
                integral_2[i, 0] = sp.integrate(int_expr_2[i],
                                                (sp.Symbol('y'), np.min(self.y), np.max(self.y)))
            f_vec += integral_2

        if np.sum(np.isclose(self.x, 0)) == 2:
            shape_func_subs = np.array([elem.subs(sp.Symbol('x'), 0) for elem in self.shape_func])
            int_expr_3 = self.thickness * self.q * shape_func_subs
            integral_3 = np.zeros_like(f_vec)
            for i in range(integral_3.shape[0]):
                integral_3[i, 0] = sp.integrate(int_expr_3[i],
                                                (sp.Symbol('y'), np.min(self.y), np.max(self.y)))
            f_vec += integral_3

            if self.verbose:
                print("Heat supply from the left.")

        return f_vec
