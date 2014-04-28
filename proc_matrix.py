import numpy as np
import json
def __calculate_median(mat):
	return np.median(mat)

def __mat_to_list(mat):
	return mat.tolist()

def __return_special_array(mat, lower_num):
	return mat[np.nonzero(mat > lower_num)]

def __return_special_indices(mat, lower_num):
	return np.nonzero(mat > lower_num)

def read_name_dict(filename):
	f = open(filename, 'r')
	name_dict = json.loads(f.read())
	f.close()
	return name_dict

def __get_doc_name_by_indice(indice, name_dict):
	return name_dict[indice]

def main(filename, name_map_filename):
	array_mat = np.load(filename)
	median = __calculate_median(__return_special_array(array_mat, 0))

	# print(array_mat)
	print(median)
	print(__return_special_array(array_mat, median))

	name_dict = read_name_dict(name_map_filename)
	indice_arr = __return_special_indices(array_mat, median)
	for x in range(len(indice_arr[0])):
		row_indice = str(indice_arr[0][x])
		col_indice = str(indice_arr[1][x])
		name1 = __get_doc_name_by_indice(row_indice, name_dict)
		name2 = __get_doc_name_by_indice(col_indice, name_dict)
		num = array_mat[row_indice, col_indice]
		print(str(name1) + ' & ' + str(name2) + ' : ' + str(num))
		

if __name__ == '__main__':
	main("cos_list.npy", "doc_name_map")



