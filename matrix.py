class Matrix:
    def __init__(self, matrix_lst):
        flag = False
        if type(matrix_lst) == list:
            if len(matrix_lst) > 1:
                for i in matrix_lst:
                    if type(i) != list:
                        flag = True
                        break
                if flag:
                    raise Exception("only list are allowed")
                flag = False
                lenx = len(matrix_lst[0])
                for i in matrix_lst:
                    if len(i) != lenx:
                        flag = True
                        break
                if flag:
                    raise Exception("no of elements in a row is not same")
                self.row = len(matrix_lst)
                self.column = len(matrix_lst[0])
                self.mat = matrix_lst
            elif len(matrix_lst) == 1:
                if type(matrix_lst[0]) == list:
                    self.row = 1
                    self.column = len(matrix_lst[0])
                    self.mat = matrix_lst
                else:
                    raise Exception("only list are allowed")
            else:
                self.row = 0
                self.column = 0
                self.mat = [[]]
            self.mat = self.mat

    def __round__(self, n=None):
        temp = self.mat
        if n is None or type(n) in Support_Object_int:
            for i in range(self.row):
                for j in range(self.column):
                    temp[i][j] = round(self.mat[i][j], n)
            return Matrix(temp)
        else:
            raise TypeError('{} is not supported in round'.format(type(n)))

    def __add__(self, other):
        if type(other) == type(self):
            if other.row == self.row:
                if other.column == self.column:
                    temp1 = []
                    for i in range(self.row):
                        temp2 = []
                        for j in range(self.column):
                            temp2.append(self.mat[i][j]+other.mat[i][j])
                        temp1.append(temp2)
                    return Matrix(temp1)
                else:
                    raise Exception('column are not same')
            else:
                raise Exception('row are not same')
        else:
            raise Exception('cannot add a matrix with {}'.format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) == type(self):
            if other.row == self.row:
                if other.column == self.column:
                    temp1 = []
                    for i in range(self.row):
                        temp2 = []
                        for j in range(self.column):
                            temp2.append(self.mat[i][j]-other.mat[i][j])
                        temp1.append(temp2)
                    return Matrix(temp1)
                else:
                    raise Exception('column are not same')
            else:
                raise Exception('row are not same')
        else:
            raise Exception('cannot subtract a matrix with {}'.format(type(other)))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __iadd__(self, other):
        if type(other) == type(self):
            if other.row == self.row:
                if other.column == self.column:
                    temp1 = []
                    for i in range(self.row):
                        temp2 = []
                        for j in range(self.column):
                            temp2.append(self.mat[i][j]+other.mat[i][j])
                        temp1.append(temp2)
                    return Matrix(temp1)
                else:
                    raise Exception('column are not same')
            else:
                raise Exception('row are not same')
        else:
            raise Exception('cannot add a matrix with {}'.format(type(other)))

    def __isub__(self, other):
        if type(other) == type(self):
            if other.row == self.row:
                if other.column == self.column:
                    temp1 = []
                    for i in range(self.row):
                        temp2 = []
                        for j in range(self.column):
                            temp2.append(self.mat[i][j]-other.mat[i][j])
                        temp1.append(temp2)
                    return Matrix(temp1)
                else:
                    raise Exception('column are not same')
            else:
                raise Exception('row are not same')
        else:
            raise Exception('cannot subtract a matrix with {}'.format(type(other)))

    def __neg__(self):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(-self.mat[i][j])
            temp1.append(temp2)
        return Matrix(temp1)

    def __truediv__(self, other):
        if type(other) in Support_Object:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j]/other)
                temp1.append(temp2)
            return Matrix(temp1)
        else:
            raise Exception('cannot divide a matrix with {}'.format(type(other)))

    def __rtruediv__(self, other):
        if type(other) in Support_Object:
            temp = self.inverse()
            temp *= other
            return temp

    def __idiv__(self, other):
        if type(other) in Support_Object:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j]/other)
                temp1.append(temp2)
            return Matrix(temp1)
        else:
            raise Exception('cannot divide a matrix with {}'.format(type(other)))

    def __floordiv__(self, other):
        if type(other) in Support_Object_real:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j]//other)
                temp1.append(temp2)
            return Matrix(temp1)
        else:
            raise Exception('cannot do a floor divide a matrix with {}'.format(type(other)))

    def __ifloordiv__(self, other):
        if type(other) in Support_Object_real:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j]//other)
                temp1.append(temp2)
            return Matrix(temp1)
        raise Exception('cannot do a floor divide a matrix with {}'.format(type(other)))

    def __mul__(self, other):
        if type(other) == type(self):
            if other.row == self.column:
                temp1 = [[0 for _ in range(other.column)] for __ in range(self.row)]
                temp = transpose(other)
                for i in range(self.row):
                    for j in range(temp.row):
                        temp2 = 0*other[0][0]
                        for k in range(self.column):
                            temp2 += self.mat[i][k]*temp.mat[j][k]
                        temp1[i][j] = temp2
                return Matrix(temp1)
            else:
                raise Exception('rows of self are not similar to column of other')
        elif type(other) in Support_Object2:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j] * other)
                temp1.append(temp2)
            return Matrix(temp1)
        else:
            raise Exception('cannot do the matrix multiplication with {}'.format(type(other)))

    def __rmul__(self, other):
        if type(other) in Support_Object2:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j] * other)
                temp1.append(temp2)
            return Matrix(temp1)
        else:
            raise Exception('cannot do the matrix multiplication with {}'.format(type(other)))

    def __imul__(self, other):
        if type(other) == type(self):
            if other.row == self.column:
                temp1 = [[0 for _ in range(other.column)] for __ in range(self.row)]
                temp = transpose(other)
                for i in range(self.row):
                    for j in range(temp.row):
                        temp2 = 0
                        for k in range(self.column):
                            temp2 += self.mat[i][k]*temp.mat[j][k]
                        temp1[i][j] = temp2
                return Matrix(temp1)
            else:
                raise Exception('rows of self are not similar to column of other')
        elif type(other) in Support_Object:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j] * other)
                temp1.append(temp2)
            return Matrix(temp1)
        else:
            raise Exception('cannot do the matrix multiplication with {}'.format(type(other)))

    def __mod__(self, other):
        if type(other) in Support_Object_real:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j] % other)
                temp1.append(temp2)
            return Matrix(temp1)
        else:
            raise Exception('cannot find a%b of a matrix with {}'.format(type(other)))

    def __imod__(self, other):
        if type(other) in Support_Object_real:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j] % other)
                temp1.append(temp2)
            return Matrix(temp1)
        else:
            raise Exception('cannot find a%b of a matrix with {}'.format(type(other)))

    def __eq__(self, other):
        if type(other) == type(self):
            return self.mat == other.mat
        else:
            Exception("couldn't check Matrix with {}".format(type(other)))

    def __ne__(self, other):
        if type(other) == type(self):
            return self.mat != other.mat
        else:
            Exception("couldn't check Matrix with {}".format(type(other)))

    def __abs__(self):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(abs(self.mat[i][j]))
            temp1.append(temp2)
        return Matrix(temp1)

    def __pow__(self, power, modulo=None):
        if type(power) in Support_Object_int:
            if self.row == self.column:
                temp = identityMatrix(self.row)
                if power < 0:
                    temp2 = self.inverse()
                    temp *= temp2
                    power = abs(power)
                    power -= 1
                else:
                    temp2 = self
                for i in range(power):
                    temp *= temp2
                return temp
            else:
                raise Exception('Could not find Matrix**{} for non square matrix'.format(power))
        else:
            raise Exception('Only these types are supported {}'.format(Support_Object_int))

    def complex(self):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(complex(self.mat[i][j]))
            temp1.append(temp2)
        return Matrix(temp1)

    def int(self):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(int(self.mat[i][j]))
            temp1.append(temp2)
        return Matrix(temp1)

    def float(self):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(float(self.mat[i][j]))
            temp1.append(temp2)
        return Matrix(temp1)

    def long(self):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append((self.mat[i][j]))
            temp1.append(temp2)
        return Matrix(temp1)

    def oct(self):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(oct(self.mat[i][j]))
            temp1.append(temp2)
        return Matrix(temp1)

    def hex(self):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(hex(self.mat[i][j]))
            temp1.append(temp2)
        return Matrix(temp1)

    def transpose(self):
        temp = [[0 for _ in range(self.row)] for __ in range(self.column)]
        for i in range(self.row):
            for j in range(self.column):
                temp[j][i] = self.mat[i][j]
        self.mat = temp
        self.row, self.column = self.column, self.row

    def mulmat(self, other):
        if type(other) == type(self):
            if other.row == self.column:
                temp1 = [[0 for _ in range(other.column)] for __ in range(self.row)]
                temp = transpose(other)
                for i in range(self.row):
                    for j in range(temp.row):
                        temp2 = 0
                        for k in range(self.column):
                            temp2 += self.mat[i][k]*temp.mat[j][k]
                        temp1[i][j] = temp2
                self.mat = temp1
                self.row = len(temp1)
                self.column = len(temp1[0])
            else:
                raise Exception('rows of self are not similar to column of other')
        elif type(other) in Support_Object:
            temp1 = []
            for i in range(self.row):
                temp2 = []
                for j in range(self.column):
                    temp2.append(self.mat[i][j] * other)
                temp1.append(temp2)
            self.mat = temp1
            self.row = len(temp1)
            self.column = len(temp1[0])
        else:
            raise Exception('cannot do the matrix multiplication with {}'.format(type(other)))

    def map(self, minPossibleVal, maxPossibleVal, minVal, maxVal):
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(map_it(self.mat[i][j], minPossibleVal, maxPossibleVal, minVal, maxVal))
            temp1.append(temp2)
        self.mat = temp1

    def det(self):
        if self.row == self.column:
            plist = list(s_permutations([i+1 for i in range(self.row)]))
            temp = 0
            for k in range(len(plist)):
                temp2 = 1
                for i in range(len(plist[k])):
                    temp2 *= self.mat[i][plist[k][i] - 1]
                if k % 2 == 0:
                    temp += temp2
                else:
                    temp -= temp2
            return temp
        else:
            raise Exception('Could not find the det of a non square matrix')

    def pivotize(self):
        n = self.row
        ID = [[float(i == j) for i in range(n)] for j in range(n)]
        r = 0
        for j in range(n):
            row = max(range(j, n), key=lambda i: abs(self.mat[i][j]))
            if j != row:
                ID[j], ID[row] = ID[row], ID[j]
                r += 1
        return Matrix(ID), r

    def ludecompose(self):
        if self.row == self.column:
            n = self.row
            L = [[0.0] * n for _ in range(n)]
            U = [[0.0] * n for _ in range(n)]
            P, r = self.pivotize()
            A2 = P*self
            for j in range(n):
                L[j][j] = 1.0
                for i in range(j + 1):
                    s1 = sum(U[k][j] * L[i][k] for k in range(i))
                    U[i][j] = A2[i][j] - s1
                for i in range(j, n):
                    s2 = sum(U[k][j] * L[i][k] for k in range(j))
                    L[i][j] = (A2[i][j] - s2) / U[j][j]
            return Matrix(L), Matrix(U), P, r
        else:
            raise TypeError('Cannot find ludecomposition matrix of non square matrix')

    def trace(self):
        r = 1
        for i in range(self.row):
            if len(self.mat[i]) <= i:
                break
            r *= self.mat[i][i]
        return r

    def det_fast(self):
        l, u, p, r = self.ludecompose()
        return (-1)**r * l.trace() * u.trace()

    def minor(self, i, j):
        if self.row == self.column:
            return Matrix([row[:j] + row[j+1:] for row in (self.mat[:i] + self.mat[i + 1:])])
        else:
            raise Exception('Minor not Possible of non Square matrix')

    def minorMatrix(self):
        temp = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(self.minor(i, j).det())
            temp.append(temp2)
        return Matrix(temp)

    def cofactor(self):
        temp = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(pow(-1, (i+1)+(j+1))*self.minor(i, j).det())
            temp.append(temp2)
        return Matrix(temp)

    def adjoint(self):
        temp = self.cofactor()
        temp.transpose()
        return temp

    def inverse(self):
        if self.det() != 0:
            return 1/self.det() * self.adjoint()
        else:
            raise Exception('Det = 0 could not find inverse')

    def addrow(self, lst, key=None):
        if type(lst) == list:
            if key is None:
                key = self.row
            if self.row >= key > -1:
                self.mat.insert(key, lst)
                self.row += 1
            else:
                raise ValueError('Matrix index out of bond')
        else:
            raise Exception('Only list is supported')

    def delrow(self, key=None):
        if key is None:
            key = self.row-1
        if self.row > key > -1:
            del self.mat[key]
            self.row -= 1
        else:
            raise ValueError('Matrix index out of bond')

    def addcolumn(self, lst, key=None):
        if type(lst) == list:
            if key is None:
                key = self.column
            if self.column >= key > -1:
                for i in range(self.row):
                    self.mat[i].insert(key, lst[i])
                self.column += 1
            else:
                raise ValueError('Matrix index out of bond')
        else:
            raise Exception('Only list is supported')

    def delcolumn(self, key=None):
        if key is None:
            key = self.column-1
        if self.row > key > -1:
            for i in range(self.row):
                del self.mat[i][key]
            self.column -= 1
        else:
            raise ValueError('Matrix index out of bond')

    def symmetric(self):
        if self.row == self.column:
            return 1/2 * (self + transpose(self))
        else:
            raise Exception('Could not find symmetric part of non square matrix')

    def skew_symmetric(self):
        if self.row == self.column:
            return 1/2 * (self - transpose(self))
        else:
            raise Exception('Could not find skew symmetric part of non square matrix')

    def __str__(self):
        biggest = self.__max + 2
        string = ''
        for i in range(self.row):
            if i < self.row-1:
                string += '{}\n'
            else:
                string += '{}'
            if self.column > 0:
                string2 = '['
            else:
                string2 = '[]'
            for j in range(self.column):
                if j == 0:
                    if self.column > 1:
                        string2 += '{:<{}}'.format(self.mat[i][j], biggest-1)
                    else:
                        string2 += '{}]'.format(self.mat[i][j], biggest - 1)
                elif j < self.column - 1:
                    string2 += '{: ^{}}'.format(self.mat[i][j], biggest)
                else:
                    string2 += '{:{}}]'.format(self.mat[i][j], biggest-1)
            string = string.format(string2)
        if len(string) > 0:
            return string
        else:
            return 'empty'

    def __getitem__(self, item):
        return self.mat[item]

    def __setitem__(self, key, value):
        if len(value) == self.column:
            self.mat[key] = value
        else:
            raise Exception('invalid change')

    def __len__(self):
        return len(self.mat)

    @property
    def __max(self):
        biggest = -float('inf')
        for i in range(self.row):
            for j in range(self.column):
                if len(str(self.mat[i][j])) > biggest:
                    biggest = len(str(self.mat[i][j]))
        return biggest


class SystemOfEquation:
    def __init__(self, A, B, X, roundof=None):
        if type(A) != Matrix and type(A) == list and type(A[0]) == list:
            A = Matrix(A)
        elif type(A) == Matrix:
            pass
        else:
            raise TypeError('invalid Matrix A')
        if type(B) != Matrix and type(B) == list and type(B[0]) == list:
            B = Matrix(B)
        elif type(B) == Matrix:
            pass
        else:
            raise TypeError('invalid Matrix B')
        if type(X) != Matrix and type(X) == list and type(X[0]) == list:
            X = Matrix(X)
        elif type(X) == Matrix:
            pass
        else:
            raise TypeError('invalid Matrix X')
        if A.det() != 0:
            if B.row == X.row and B.column == X.column and A.column == B.row:
                if roundof is None:
                    self.answer = A.inverse()*B
                else:
                    self.answer = round(A.inverse()*B, roundof)
                temp = []
                for i in range(self.answer.row):
                    temp2 = []
                    for j in range(self.answer.column):
                        temp2.append('{} = {}'.format(X[i][j], self.answer[i][j]))
                    temp.append(temp2)
                self.better_ans = Matrix(temp)
            else:
                raise TypeError('invalid equation')
        else:
            raise Exception('this system of equation has many solution')

    def __str__(self):
        return str(self.better_ans)


def identityMatrix(size):
    temp = []
    for i in range(size):
        temp2 = []
        for j in range(size):
            if i == j:
                temp2.append(1)
            else:
                temp2.append(0)
        temp.append(temp2)
    return Matrix(temp)


def zeroMatrix(size):
    temp = [[0 for _ in range(size)] for _ in range(size)]
    return Matrix(temp)


def singularMatrxi(element):
    return Matrix([[element]])


def transpose(self):
    if type(self) == Matrix:
        temp = [[0 for _ in range(self.row)] for __ in range(self.column)]
        for i in range(self.row):
            for j in range(self.column):
                temp[j][i] = self.mat[i][j]
        return Matrix(temp)
    else:
        raise Exception('cannot find transpose of {}'.format(type(self)))


def map_it(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


def mapMat(self, minPossibleVal, maxPossibleVal, minVal, maxVal):
    if type(self) == Matrix:
        temp1 = []
        for i in range(self.row):
            temp2 = []
            for j in range(self.column):
                temp2.append(map_it(self.mat[i][j], minPossibleVal, maxPossibleVal, minVal, maxVal))
            temp1.append(temp2)
        return Matrix(temp1)
    else:
        raise Exception('cannot do map(matrix) operation on {}'.format(type(self)))


def s_permutations(seq):
    items = [[]]
    new_items = []
    for j in seq:
        new_items = []
        for i, item in enumerate(items):
            if i % 2:
                new_items += [item[:i] + [j] + item[i:]
                              for i in range(len(item) + 1)]
            else:
                new_items += [item[:i] + [j] + item[i:]
                              for i in range(len(item), -1, -1)]
        items = new_items

    return new_items


Support_Object = [float, int, complex, Matrix]
Support_Object2 = [float, int, complex]
Support_Object_real = [float, int]
Support_Object_int = [int]
