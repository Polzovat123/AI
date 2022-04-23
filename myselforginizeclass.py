import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

class SOMNetwork():
	def __init__(self, input_size, W=10, H=10, sigma=None, learning_rate=0.1, max_iter=1000, dtype=tf.float64):
		if not sigma:
			# Начальное значение радиуса окрестности BMU  (sigma0)
			sigma = max(W, H) / 2
		self.dtype = dtype
		self.max_iter = max_iter
		# Определение констант tensorflow
		self.W = tf.constant(W, dtype=tf.int64)		# размерность сети по ширине
		self.H = tf.constant(H, dtype=tf.int64)		# размерность сети по высоте
		self.learning_rate0 = tf.constant(learning_rate, dtype=dtype, name='learning_rate')	# начальная скорость обучения
		self.min_learning_rate = tf.constant(0.01, dtype=self.dtype, name='min_learning_rate')	# минимально допустимая скорость обучения
		self.sigma0 = tf.constant(sigma, dtype=dtype, name='sigma')	# начальный радиус окрестности BMU
		self.lambdaT = tf.constant(max_iter/np.log(sigma), dtype=dtype, name='lambdaT')	# постоянная времени
		self.min_sigma = tf.constant(sigma * np.exp(-max_iter/(max_iter/np.log(sigma))), dtype=dtype, name='min_sigma')
		self.max_iter = tf.constant(max_iter, dtype=dtype, name='max_iter')		# Максимальное количество итераций
		# Контейнер для входного вектора
		self.x = tf.compat.v1.placeholder(shape=[input_size], dtype=dtype, name='input')
		# Контейнер для номера текущей итерации
		self.iter = tf.compat.v1.placeholder(dtype=dtype, name='iteration')
		# начальная инициализация весов нейронов случайными числами
		self.w = tf.Variable(tf.random.uniform([W*H, input_size], minval=-1, maxval=1, dtype=dtype),
			dtype=dtype, name='weights')
		# Матрица координат нейронов
		self.positions = tf.where(tf.fill([H, W], True))

	# Возвращает координаты BMU на текущей итерации
	def get_bmu_loc(self):
		return self.cur_bmu_location

	# Обучение сети
	def train(self):
		# определяем BMU (центроид) для входного вектора по евклидовому расстоянию
		win_index = self.competition()
		# Определяем текущие координаты BMU на нашей карте по индексу
		win_loc_2d = self.get_loc_by_index(win_index)
		self.cur_bmu_location = win_loc_2d
		# Определяем квадрат евклидового расстояния между BMU и другими нейронами
		squared_distance = tf.reduce_sum(tf.square(tf.cast(self.positions - tf.dtypes.cast(win_loc_2d, tf.int64), dtype=self.dtype)), axis=1)
		# Определяем радиус окрестности BMU
		#sigma = tf.cond(self.iter > self.max_iter, lambda: self.min_sigma, lambda: self.sigma0 * tf.exp(-self.iter/self.lambdaT))
		sigma = self.sigma0 * tf.exp(-self.iter/self.lambdaT)
		# Вычисляем латеральное расстояние (lateral distance)
		theta = tf.exp(-squared_distance / (2 * tf.square(sigma)))
		# Определяем скорость обучения
		learning_rate = self.learning_rate0 * tf.exp(-self.iter/self.max_iter)
		learning_rate = tf.cond(learning_rate <= self.min_learning_rate, lambda: self.min_learning_rate, lambda: learning_rate)
		# корректировка весов
		delta = tf.transpose(learning_rate * theta * tf.transpose(self.x - self.w))
		updating = tf.compat.v1.assign(self.w, self.w + delta)
		#delta = tf.cond(squared_distance < tf.square(sigma), lambda: self.weights_adjusting(squared_distance, sigma), lambda: tf.zeros(self.w.shape, tf.float64))
		#updating = tf.compat.v1.assign(self.w, self.w + delta)
		return updating

	# Определяем индекс BMU по минимальному евклидовому расстоянию
	def competition(self, info=''):
		distance = tf.sqrt(tf.reduce_sum(tf.square(self.x - self.w), axis=1))
		return tf.argmin(distance, axis=0)

	# вычисляем 2D-координаты по индексу (порядковому номеру элемента)
	def get_loc_by_index(self, ind):
		row = ind // self.W
		col = ind - row * self.W
		return [row, col]

def heatmap2d(arr: np.ndarray):
	plt.imshow(arr, cmap='viridis')
	plt.title('Clusters map')
	plt.colorbar()
	plt.show()

# Вывод самоорганизующейся карты с помощью цветовой палитры
def get_som_with_color_data(color_data, map_width, map_height, max_iter, input_size):
	# Создаем объект SOM
	som = SOMNetwork(input_size=input_size, W=map_width, H=map_height, dtype=tf.float64)
	training = som.train()
	init = tf.compat.v1.global_variables_initializer()
	# Запускаем вычислительную сессию (Session) tensorflow
	with tf.compat.v1.Session() as sess:
		init.run()
		img1 = tf.reshape(som.w, [map_height, map_width, -1]).eval()
		plt.figure(1)
		plt.subplot(121)
		plt.imshow(img1)
		start = time.time()
		for i, data_row in enumerate(color_data):
			if i % max_iter == 0:
				print('Epoch:', i/max_iter)
			sess.run(training, feed_dict={ som.x: data_row, som.iter: i })
		end = time.time()
		print(end - start)
		img2 = tf.reshape(som.w, [map_height, map_width, -1]).eval()
		plt.subplot(122)
		plt.imshow(img2)
	plt.show()

# Кластеризация набора данных
def get_som(data, map_width, map_height, max_iter, input_size):
	cindexes = []  # индексы кластеров
	cindex = 0
	# Создаем объект SOM
	som = SOMNetwork(input_size=input_size, W=map_width, H=map_height, dtype=tf.float64)
	training = som.train()
	clocations = []		# список для сохранения позиций BMU в процессе самоорганизации сети
	init = tf.compat.v1.global_variables_initializer()
	# Запускаем вычислительную сессию (Session) tensorflow
	with tf.compat.v1.Session() as sess:
		init.run()
		plt.figure(1)
		plt.subplot(121)
		start = time.time()
		for i, data_row in enumerate(data):
			if i % max_iter == 0:
				print('Epoch:', i/max_iter)
			sess.run(training, feed_dict={som.x: data_row, som.iter: i})
			# сохраняем координаты BMU, полученные в процессе самоорганизации сети
			clocation = sess.run(som.get_bmu_loc(), feed_dict={ som.x: data_row })
			clocations.append(clocation)
		end = time.time()
		print(end - start)
		# Формируем список уникальных (неповторяющихся) координат BMU
		# и подсчитываем количество элементов в кластере, который определяется данным BMU (центроидом)
		clocations_uniq = []
		rows_in_cluster = np.zeros((map_height, map_width)).astype(int)	# numpy массив для отображения карты кластеров
		# Список кластеров для добавления в конечный файл
		cindexes = np.zeros(len(clocations)).astype(int)
		# Массив индексов (для визуализации)
		ind_arr = np.arange(map_width*map_height).reshape(rows_in_cluster.shape)
		for i, loc in enumerate(clocations):
			if loc not in clocations_uniq:
				clocations_uniq.append(loc)
				rows_in_cluster[loc[0], loc[1]] = 1
			else:
				rows_in_cluster[loc[0], loc[1]] += 1
			cindexes[i] = ind_arr[loc[0], loc[1]]
		cnums = list(dict.fromkeys(cindexes))	# уникальные индексы центроидов
		# Отображаем полученные центроиды на координатной плоскости
		for i, loc in enumerate(clocations_uniq):
			plt.scatter(loc[1], loc[0], c='tab:blue', alpha=0.5, edgecolors='none')
			plt.annotate(f'{cnums[i]}_({rows_in_cluster[loc[0], loc[1]]})', (loc[1], loc[0]), fontsize=6)
		# Отображаем цветовую карту по количеству элементов в кластерах
		plt.subplot(122)
		plt.title('Clusters map')
		plt.imshow(np.flipud(rows_in_cluster), cmap='viridis')
		plt.colorbar()
	plt.show()
	# Возвращаем индексы кластеров для dataset
	return cindexes

if __name__ == '__main__':
	# Режим выполнения вычислений v1
	tf.compat.v1.disable_eager_execution()
	# Загрузка набора данных из файла
	dataset = pd.read_csv("data.csv", sep=';')
	pd.set_option("display.max_rows", None, "display.max_columns", None)  # Настройки вывода таблицы

	# Сохранем идентификаторы записей
	labels = dataset['Код']

	# Удаляем несущественые для кластеризации столбцы
	dataset.drop(['Код'], axis=1, inplace=True)
	# Помещаем названия колонок в отдельный список
	feature_cols = list(dataset.columns)
	print(dataset.shape)  # Выводим размерность данных
	print(dataset.head(5)) # Выводим первые 5 строк DataFrame (проверяем, что файл загрузился верно)

	# Выбираем признаки для one-hot кодирования
	# (раскомментировать, если в данных есть значимые категории для One-Hot кодирования)
	"""
	one_hot_cols = ['col_name1', 'col_name2', 'col_name3']
	for col_name in one_hot_cols:
		one_hot = pd.get_dummies(dataset[col_name])
		dataset = dataset.drop(col_name, axis=1)
		dataset = dataset.join(one_hot)
	"""

	# Удаляем пропуски в наборе данных (если необходимо)
	for col_name in feature_cols:
		removed = dataset.drop(dataset.loc[dataset[col_name].isna(), [col_name]].index, inplace=True)
		print(f'Rows in {col_name} removed: ', removed)

	# Нормализация данных - линейная нормализация Xnorm = (Xi - Xmin)/(Xmax - Xmin)) -> [0...1]
	norm = MinMaxScaler()       # Нормализатор для входных данных
	# Делаем MinMax-нормализацию входных значений
	in_data = norm.fit_transform(dataset.values)
	print("Размерность входных данных: ", in_data.shape)

	# Формирование цветовых маркеров случайным образом (для тестирования)
	#test_color_data = np.random.uniform(0, 1, (25000, 3))
	# Кластеризация по зветовой палитре
	#get_som_with_color_data(test_color_data, map_width=30, map_height=30, max_iter=1000, input_size=3)

	# Запускаем кластеризацию и возвращаем номера кластеров для набора данных
	cindexes = get_som(in_data, map_width=16, map_height=12, max_iter=1000, input_size=10)

	dataset['Код'] = labels		# добавляем сохраненную ранее информационную колонку (если необходимо)
	# Добавляем номера кластеров в отдельную колонку набора данных
	dataset['Cluster'] = cindexes
	# Сортируем строки в наборе данных по номерам кластеров
	dataset.sort_values(by=['Cluster'], inplace=True)
	dataset.to_csv('Clusterized.csv', index=True, sep=";", encoding="utf-8-sig")  # сохраняем в файл Clusterized.csv
