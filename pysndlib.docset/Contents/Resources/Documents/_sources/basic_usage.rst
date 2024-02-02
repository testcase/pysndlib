Basic Usage
=========================

A convenient way to get started with pysndlib is to use the Sound context manager to handle 
manager a number of details when rendering audio to a file or numpy array.

* opening a closing of the main soundfile being written to including giving it a name
* setting the samplerate, header type and sample type of the soundfile 
* scaling of the audio values to a specific range



the following will generate a sound file with two sine waves
if will play after rendering using CLM.player
clipped is False which means it does not clip the signal at -1 to 1
scaled_to with normalize amplitude to .5
if clipped was True (the default) the signal would be clipped first and then scaled
which gives a very different output in this example

::

    import pysndlib.clm as clm

    with clm.Sound(play=True, statistics=True, scaled_to=.5, clipped=False):
        osc1 = clm.make_oscil(330)
        osc2 = clm.make_oscil(500)
        for i in range(44100):
            clm.outa(i, clm.oscil(osc1) + clm.oscil(osc2))


Here is an example showing writing the same thing to a numpy array
and using finalize to call a function to plot the signal

::

    import pysndlib.clm as clm
    from matplotlib import pyplot as plt

    def plot_mono_arr(arr):
        plt.figure(figsize=(6,6))
        plt.plot(arr[0], color='gray')
        plt.tight_layout()
        plt.show()


    outarr = np.zeros((1,44100))


    with clm.Sound(outarr, statistics=True, scaled_to=.5, clipped=False, finalize=plot_mono_arr):
        osc1 = clm.make_oscil(330)
        osc2 = clm.make_oscil(500)
        for i in range(44100):
            clm.outa(i, clm.oscil(osc1) + clm.oscil(osc2))


This is an example of defining a reverb function that is applied after the initial output is
rendered. using locsig makes this even easier. to make a reverb you must also use the @clm.clm_reverb decorator

Note clm.default.reverb gets defined within the with Sound context.

::
    
    @clm.clm_reverb
    def bad_mono_rev1(volume=1.):
        ap1 = clm.make_all_pass(-.9, .9, 1051)
        ap2 = clm.make_all_pass(-.9, .9, 1207)
        ap3 = clm.make_all_pass(-.9, .9, 1000)
        dly = clm.make_delay(clm.seconds2samples(.011))
        length = clm.get_length(clm.default.reverb)
    
        apb = clm.make_all_pass_bank([ap1, ap2, ap3])
    
        for i in range(length):
            clm.outa(i, clm.delay(dly, volume * clm.all_pass_bank(apb, clm.ina(i, clm.default.reverb))))

    with clm.Sound('ex2.wav', play=True, statistics=True, reverb=bad_mono_rev1(.5)):
        osc = clm.make_oscil(500)
        e = clm.make_env([0.,0.0, .05, .8, .4, 0.0, 1.0, 0.0], length=22050)
        for i in range(44100*2):
            val = clm.env(e)*clm.oscil(osc)
            clm.outa(i, val*.6)
            clm.outa(i, val*.4, clm.default.reverb)



Lastly an example that demonstrates defining functions that can be called in Sound context
that uses finalize to plot the output file after the reverb has been applied

::

    from matplotlib import pyplot as plt

    def plot_mono_soundfile(filename):
        y, _ = file2ndarray(filename)
        plt.figure(figsize=(6,6))
        plt.plot(y[0], color='gray')
        plt.tight_layout()
        plt.show()


    @clm.clm_reverb
    def bad_mono_rev1(volume=1.):
        ap1 = clm.make_all_pass(-.9, .9, 1051)
        ap2 = clm.make_all_pass(-.9, .9, 1207)
        ap3 = clm.make_all_pass(-.9, .9, 1000)
        dly = clm.make_delay(clm.seconds2samples(.011))
        length = clm.get_length(clm.default.reverb)
    
        apb = clm.make_all_pass_bank([ap1, ap2, ap3])
    
        for i in range(length):
            clm.outa(i, clm.delay(dly, volume * clm.all_pass_bank(apb, clm.ina(i, clm.default.reverb))))
        
        
    def blip(start, dur, freq):
        osc = clm.make_oscil(freq)
        beg = clm.seconds2samples(start)
        end = beg + clm.seconds2samples(dur)
        e = clm.make_env([0.,0.0, .05, .8, .4, 0.0, 1.0, 0.0], duration=dur*.5)
        for i in range(beg, end):
            val = clm.oscil(osc) * env(e)
            clm.outa(i, val*.6)
            clm.outa(i, val*.4, clm.default.reverb)


    with clm.Sound('ex2.wav', play=True, statistics=True, reverb=bad_mono_rev1, finalize=plot_mono_soundfile):
        blip(0, 1, 400)
        blip(1, 1, 500)
        blip(2, 1, 600)
        blip(3, 1, 900)
        for i in np.arange(4, 6, .333333):
            blip(i, .5, 800)



