import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from scipy.interpolate import interp1d 
from scipy.signal import butter, lfilter
from scipy import signal

measures = {} #A dictionary to store all the values 
def get_data(filename):
    dataset = pd.read_csv(filename)  #Read data from CSV datafile
    return dataset

#Define the filter
def butter_lowpass(cutoff, fs, order=5): #Calculate the constants a, b for the Butterworth filter
    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def rolmean(dataset, hrw, fs):
    mov_avg = dataset.hart.rolling(window=int(hrw*fs)).mean() #Calculate moving average 
    #Fill in mov_avg with the plain value where moving average function returns NaN, which is at the beginning of the signal 
    avg_hr = (np.mean(dataset.hart))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    #mov_avg = [x*1.2 for x in mov_avg] #Raise the average by 20% to prevent the secondary heart contraction from interfering
    dataset['hart_rollingmean'] = mov_avg #Append the moving average to the dataframe

def detect_peaks(dataset, ma_perc, fs): #ma_perc = moving average percentage (dynamic threshold used for detecting peaks correctly in noisy signals)
    rolmean = [(x+((x/100)*ma_perc)) for x in dataset.hart_rollingmean] #Raise moving average with passed ma_perc
    window = []
    peaklist = []
    listpos = 0
    for datapoint in dataset.hart: #Mark ROIs (Regions Of Interest)
        rollingmean = rolmean[listpos]
        if (datapoint <= rollingmean) and (len(window) <= 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    measures['peaklist'] = peaklist
    measures['ybeat'] = [dataset.hart[x] for x in peaklist]
    measures['rolmean'] = rolmean
    calc_RR(dataset, fs)
    measures['rrsd'] = np.std(measures['RR_list'])

def fit_peaks(dataset, fs):
    ma_perc_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] #List with moving average raise percentages (could be altered in order to be more precise)
    rrsd = []
    valid_ma = []
    for x in ma_perc_list: #Detect peaks with all percentages, append results to list 'rrsd'
        detect_peaks(dataset, x, fs)
        bpm = ((len(measures['peaklist'])/(len(dataset.hart)/fs))*60)
        rrsd.append([measures['rrsd'], bpm, x])
    for a,b,c in rrsd: #Test list entries and select valid measures, assuming that all ECGs are taken at resting position
        if ((a > 1) and ((b > 30) and (b < 130))):
            valid_ma.append([a, c])
    measures['best'] = min(valid_ma, key = lambda t: t[0])[1] #Save the best ma_perc
    detect_peaks(dataset, min(valid_ma, key = lambda t: t[0])[1], fs) #Detect peaks with 'ma_perc' that leads to lowest rrsd
    return measures['best']

def calc_sample_rate(dataset): #Calculate Sampling Frequency fs
    sampletimer = [x for x in dataset.timer] #dataset.timer is a ms counter which starts recording at '0'
    measures['fs'] = ((len(sampletimer) / sampletimer[-1])*1000) #Divide total length of the dataset by last timer entry (= duration of ECG in ms). This is in ms, so multiply by 1000 to get the value in Hz
    fs = measures['fs'] 
    return fs

def calc_RR(dataset, fs): #Calculating heart rate
    RR_list = []
    peaklist = measures['peaklist']
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
        RR_list.append(ms_dist) #Append to list
        cnt += 1
    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list)-1)): #Calculate RR diferences
        RR_diff.append(abs(RR_list[cnt] - RR_list[cnt+1]))
        RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt+1], 2))
        cnt += 1
    measures['RR_list'] = RR_list
    measures['RR_diff'] = RR_diff
    measures['RR_sqdiff'] = RR_sqdiff

def calc_ts_measures(): #Calculate values BPM, IBI, SDNN, SDSD, RMSSD, PNN20 and PNN50
    RR_list = measures['RR_list']
    RR_diff = measures['RR_diff']
    RR_sqdiff = measures['RR_sqdiff']
    measures['bpm'] = 60000 / np.mean(RR_list)
    measures['ibi'] = np.mean(RR_list)
    measures['sdnn'] = np.std(RR_list)
    measures['sdsd'] = np.std(RR_diff)
    measures['rmssd'] = np.sqrt(np.mean(RR_sqdiff))
    NN20 = [x for x in RR_diff if (x>20)]
    NN50 = [x for x in RR_diff if (x>50)]
    measures['nn20'] = NN20
    measures['nn50'] = NN50
    measures['pnn20'] = float(len(NN20)) / float(len(RR_diff))
    measures['pnn50'] = float(len(NN50)) / float(len(RR_diff))
    #Print values
    BPM = measures['bpm']
    print("BPM = " , BPM)
    IBI = measures['ibi']
    print("IBI = ", IBI)
    SDNN = measures['sdnn']
    print("SDNN = ", SDNN)
    SDSD = measures['sdsd']
    print("SDSD = ", SDSD)
    RMSSD = measures['rmssd']
    print("RMSSD = ", RMSSD)
    PNN20 = measures['pnn20']
    print("PNN20 = ", PNN20)
    PNN50 = measures['pnn50']
    print("PNN50 = ", PNN50)

def process(dataset, hrw, fs): #Main function which calls the other functions 
    rolmean(dataset, hrw, fs)
    valid_ma_perc = fit_peaks(dataset, fs) #Calculate the best-fit moving average percentage
    detect_peaks(dataset, valid_ma_perc, fs) 
    calc_RR(dataset, fs)
    calc_ts_measures()
    plotter(dataset, "Heartbeat Plot")
    peak_rejection(dataset)
    fast_fourier(dataset, fs)

def plotter(dataset, title): 
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    plt.title(title)
    plt.plot(dataset.hart, alpha=0.5, color='blue') #Plot semi-transparent HR
    plt.plot(dataset.hart_rollingmean, color ='green', label="moving average") #Plot moving average
    plt.scatter(peaklist, ybeat, color='red', label="average: %.1f BPM" %measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.show()

def fast_fourier(dataset, fs):
    peaklist = measures['peaklist'] #Retrieve the lists needed for the transformation
    RR_list = measures['RR_list']
    RR_x = peaklist[1:] #Remove the first entry, because first interval is assigned to the second beat
    RR_y = RR_list #Y-values are equal to interval lengths
    RR_x_new = np.linspace(RR_x[0],RR_x[-1],RR_x[-1]) #Create an evenly spaced timeline starting at the second peak (RR_x[0]=peaklist[1]), ending at its endpoint and with length equal to the position of its last peak
    f = interp1d(RR_x, RR_y, kind='cubic') #Interpolate the signal with cubic spline interpolation
    plt.title("Original and Interpolated Signal")
    plt.plot(RR_x, RR_y, label="Original", color='blue')
    plt.plot(RR_x_new, f(RR_x_new), label="Interpolated", color='red')
    plt.legend()
    plt.show()
    #Set variables
    n = len(dataset.hart) #Length of the signal
    frq = np.fft.fftfreq(len(dataset.hart), d=((1/fs))) #divide the bins into frequency categories
    frq = frq[range(int(n/2))] #Get single side of the frequency range
    #Do FFT
    Y = np.fft.fft(f(RR_x_new))/n #Calculate FFT
    Y = Y[range(int(n/2))] #Return one side of the FFT
    #Plot
    plt.title("Frequency Spectrum of Heart Rate Variability")
    plt.xlim(0,0.6) #Limit X axis to frequencies of interest (0-0.6Hz for visibility, we are interested in 0.04-0.5)
    plt.ylim(0, 50) #Limit Y axis for visibility
    plt.plot(frq, abs(Y)) #Plot it
    plt.xlabel("Frequencies in Hz")
    plt.show()
    lf = np.trapz(abs(Y[(frq>=0.04) & (frq<=0.15)])) #Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the area
    print("LF:", lf)
    hf = np.trapz(abs(Y[(frq>=0.16) & (frq<=0.5)])) #Do the same for 0.16-0.5Hz (=HF)
    print("HF:", hf)    
    #Plot both filtered and unfiltered signals
    filtered = butter_lowpass_filter(dataset.hart, 2.5, fs, 5) #Filter the signal with a cutoff at 2.5Hz and a 5th order Butterworth filter
    plt.subplot(211)
    plt.plot(dataset.hart, color='Blue', alpha=0.5, label='Original Signal')
    plt.legend(loc=4)
    plt.subplot(212)
    plt.plot(filtered, color='Red', label='Filtered Signal')
    plt.ylim(200,800) #Limit filtered signal to have same y-axis as original (filter response starts at 0 so otherwise the plot will be scaled)
    plt.legend(loc=4)
    plt.show()

def peak_rejection(dataset):
    RR_list = measures['RR_list'] 
    RR_list_detrended = signal.detrend(RR_list, type='linear')
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    upper_threshold = (np.mean(RR_list) + 300) #Set thresholds
    lower_threshold = (np.mean(RR_list) - 300)
    #Detect outliers
    cnt = 0
    removed_beats = []
    removed_beats_y = []
    RR2 = []
    while cnt < len(RR_list):
        if (RR_list[cnt] < upper_threshold) and (RR_list[cnt] > lower_threshold):
            RR2.append(RR_list[cnt])
            cnt += 1
        else:
            removed_beats.append(peaklist[cnt])
            removed_beats_y.append(ybeat[cnt])
            cnt += 1
    measures['RR_list_cor'] = RR2 #Append corrected RR-list to dictionary
    plt.subplot(211)
    plt.title('Marked Uncertain Peaks')
    plt.plot(dataset.hart, color='blue', alpha=0.6, label='heart rate signal')
    plt.plot(measures['rolmean'], color='green')
    plt.scatter(measures['peaklist'], measures['ybeat'], color='green')
    plt.scatter(removed_beats, removed_beats_y, color='red', label='Detection uncertain')
    plt.legend(framealpha=0.6, loc=4)
    plt.subplot(212)
    plt.title("RR-intervals with thresholds")
    plt.plot(RR_list)
    plt.axhline(y=upper_threshold, color='red')
    plt.axhline(y=lower_threshold, color='red')
    plt.show()




