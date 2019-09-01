
import skimage.data

def conv(img, conv_filter):
	if len(img.shape) > 2 or len(conv_filter.shape) > 3:
		if img.shape[-1] != conv_filter.shape[-1]:
			print("Error Number of channels in both image and filter must match.")
			sys.exit()
	if conv_filter.shape[1] != conv_filter.shape[2]:
		print("Error: Filter must be a square matrix")
	if conv_filter.shape[1]%2==0:
		print("Error: Filter must have an odd size.")
	#An empty feature map to hold the output of convolving the filter(s) with the image.
	feature_maps = numpy.zeros((img.shape[0]-conv_filter.shape[1]+1,
								img.shape[1]-conv_filter.shape[1]+1,
								conv_filter.shape[0]))

	print("conv_filter.shape[-1]",conv_filter.shape[-1])
	print("conv_filter.shape[0]", conv_filter.shape[0])
	print("conv_filter.shape[1]", conv_filter.shape[1])
	print("conv_filter.shape[2]", conv_filter.shape[2])

	print("img.shape[-1]", img.shape[-1])
	print("img.shape[0]",  img.shape[0])
	print("img.shape[1]",  img.shape[1])
	print("img.shape[2]",  img.shape[2])

	for filter_num in range(conv_filter.shape[0]):
		print("Filter ", filter_num+1)
		curr_filter = conv_filter[filter_num, :] # getting a filter from the bank

		#checking if there are multiple channels for the single filter.
		# if so, then each channel will convolve the image
		# the result of all convolutions is summed to return a single feature map

		if len(curr_filter.shape) > 2:
			conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) #Array  holding the sum if all feature maps
			for ch_num in range(1, curr_filter.shape[-1]): #Convolving each channel with the image and summing the results.
				conv_map = conv_map + conv_img(img[:, :, ch_num], curr_filter[:, :, ch_num])
		else: # There is just a single a single channel in the filter.
			conv_map  = conv_(img, curr_filter)
		feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.

	return feature_maps




#img = skimage.data.chelsea()
img = skimage.data.imread("/home/ylebras/Documents/tools/simple_NN_C/kitten.jpg")

img = skimage.color.rgb2gray(img)

#preparing filters
l1_filter = numpy.zeros((2,3,3))

l1_filter[0, :, :] = numpy.array([[[-1, 0, 1],
								   [-1, 0 , 1],
								   [-1, 0 , 1]]])
l1_filter[1, :, :] = numpy.array([[ [1,  1,  1],
								    [0,  0,  0],
								   [-1, -1, -1]]])

l1_feature_map = conv(img, l1_filter)


