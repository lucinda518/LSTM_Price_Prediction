import lstm
import time
import matplotlib.pyplot as plt

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
	global_start_time = time.time()
	epochs  =50
	seq_len = 50
	
	print('> Loading data... ')

	orig_data, X_train, y_train, X_test, y_test = lstm.load_data('hu.csv', seq_len, True)

	print('> Data Loaded. Compiling...')

	model = lstm.build_model([1, 400,400, 1])

	hist = model.fit(
	    X_train,
	    y_train,
	    batch_size=128,
	    nb_epoch=epochs,
	    validation_split=0.05)


	plt.plot(hist.history['loss'], label = 'train')
	plt.plot(hist.history['val_loss'], label = 'test')
	plt.legend()

	predicted_model1_train = lstm.predict_point_by_point(model, X_train)
	predicted = lstm.predict_point_by_point(model, X_test)
    ## denomalize data
	predicted_model1_train_price = (predicted_model1_train+1)*orig_data[:len(predicted_model1_train)]
	predicted_price = (predicted+1)*orig_data[len(predicted_model1_train):-seq_len]
    
	print('Training duration (s) : ', time.time() - global_start_time)
	y_test_price = orig_data[len(predicted_model1_train)+seq_len:]

    
	lag = 1

	[X_train_2, y_train_2, X_test_2, y_test_2] = lstm.load_data_2(orig_data, X_train, X_test, seq_len, lag, predicted, predicted_model1_train)

	print('> Data_2 Loaded. Compiling...')

	model_2 = lstm.build_model([1, 400, 400, 1])

	hist_2 = model_2.fit(
	    X_train_2,
	    y_train_2,
	    batch_size=128,
	    nb_epoch=epochs,
	    validation_split=0.05)


	plt.plot(hist_2.history['loss'], label = 'train')
	plt.plot(hist_2.history['val_loss'], label = 'test')
	plt.legend()


	predicted_model2_train = lstm.predict_point_by_point(model_2, X_train_2)
	predicted_2 = lstm.predict_point_by_point(model_2, X_test_2)
    
	## denomalize data
	predicted_model2_train_price = (predicted_model2_train+1)*orig_data[lag:len(predicted_model1_train)]
	predicted_2_price = (predicted_2+1)*orig_data[(len(predicted_model1_train)+lag):-seq_len]
	print('Training duration (s) : ', time.time() - global_start_time)


	y_test_2_price = orig_data[len(predicted_model1_train)+seq_len+lag:]


	
	lag_2 = 2
	[X_train_3, y_train_3, X_test_3, y_test_3] = lstm.load_data_2(orig_data, X_train_2, X_test_2,  seq_len, lag_2, predicted_2, predicted_model2_train)

	print('> Data Loaded. Compiling...')
	
	model_3 = lstm.build_model([1, 400, 400, 1])

	hist_3 = model_3.fit(
	    X_train_3,
	    y_train_3,
	    batch_size=128,
	    nb_epoch=epochs,
	    validation_split=0.05)


	plt.plot(hist_3.history['loss'], label = 'train')
	plt.plot(hist_3.history['val_loss'], label = 'test')
	plt.legend()


	predicted_model3_train = lstm.predict_point_by_point(model_3, X_train_3)
	predicted_3 = lstm.predict_point_by_point(model_3, X_test_3)
    
	## denomalize data
	predicted_model3_train_price = (predicted_model3_train+1)*orig_data[lag_2:len(predicted_model1_train)]
	predicted_3_price = (predicted_3+1)*orig_data[(len(predicted_model1_train)+lag_2):-seq_len]
	print('Training duration (s) : ', time.time() - global_start_time)
    
	y_test_3_price = orig_data[len(predicted_model1_train)+seq_len+lag_2:]
	
   

	lag_3 = 3
	[X_train_4, y_train_4, X_test_4, y_test_4] = lstm.load_data_2(orig_data, X_train_3,  X_test_3, seq_len, lag_3,  predicted_3,  predicted_model3_train)

	print('> Data Loaded. Compiling...')

	model_4 = lstm.build_model([1, 400, 400, 1])

	hist_4 = model_4.fit(
	    X_train_4,
	    y_train_4,
	    batch_size=128,
	    nb_epoch=epochs,
	    validation_split=0.05)


	plt.plot(hist_4.history['loss'], label = 'train')
	plt.plot(hist_4.history['val_loss'], label = 'test')
	plt.legend()

	
	predicted_model4_train = lstm.predict_point_by_point(model_4, X_train_4)
	predicted_4 = lstm.predict_point_by_point(model_4, X_test_4)
    
	## denomalize data
	predicted_model4_train_price = (predicted_model4_train+1)*orig_data[(lag_3):len(predicted_model1_train)]
	predicted_4_price = (predicted_4+1)*orig_data[(len(predicted_model1_train)+lag_3):-seq_len]
    
    
	y_test_4_price = orig_data[len(predicted_model1_train)+seq_len+lag_3:]
	
    
	
    
	lag_4 = 4
	
	[X_train_5, y_train_5, X_test_5, y_test_5] = lstm.load_data_2(orig_data, X_train_4,  X_test_4,  seq_len, lag_4,  predicted_4,  predicted_model4_train)

	print('> Data_2 Loaded. Compiling...')

	model_5 = lstm.build_model([1, 400, 400, 1])

	hist_5 = model_5.fit(
	    X_train_5,
	    y_train_5,
	    batch_size=128,
	    nb_epoch=epochs,
	    validation_split=0.05)


	plt.plot(hist_5.history['loss'], label = 'train')
	plt.plot(hist_5.history['val_loss'], label = 'test')
	plt.legend()

	
	predicted_model5_train = lstm.predict_point_by_point(model, X_train_5)
	predicted_5 = lstm.predict_point_by_point(model_5, X_test_5)
    
	## denomalize data
	predicted_model5_train_price = (predicted_model5_train+1)*orig_data[lag_4:len(predicted_model1_train)]
	predicted_5_price = (predicted_5+1)*orig_data[(len(predicted_model1_train)+lag_4):-seq_len]	
    
	y_test_5_price = orig_data[len(predicted_model1_train)+seq_len+lag_4:]
	
	print('Training duration (s) : ', time.time() - global_start_time)

	plot_results(predicted_price, y_test_price)
	plot_results(predicted_2_price, y_test_2_price)
	plot_results(predicted_3_price, y_test_3_price)
	plot_results(predicted_4_price, y_test_4_price)
	plot_results(predicted_5_price, y_test_5_price)
    
