Sound Context Manager
==========================


.. module:: 
    pysndlib.clm
    :noindex:

**in pysndlib.clm**

.. autoclass:: Sound
    :members:



the following will generate a sound file with two sine waves
if will play after rendering using CLM.player
clipped is False which means it does not clip the signal at -1 to 1
scaled_to with normalize amplitude to .5
if clipped was True (the default) the signal would be clipped first and then scaled
which gives a very different output in this example

::

    with Sound(play=True, statistics=True, scaled_to=.5, clipped=False):
        osc1 = make_oscil(330)
        osc2 = make_oscil(500)
        for i in range(44100):
            outa(i, oscil(osc1) + oscil(osc2))


Here is an example showing writing the same thing to a numpy array
and using finalize to call a function to plot the signal

::

    from matplotlib import pyplot as plt

    def plot_mono_arr(arr):
        plt.figure(figsize=(6,6))
        plt.plot(arr[0], color='gray')
        plt.tight_layout()
        plt.show()


    outarr = np.zeros((1,44100))


    with Sound(outarr, statistics=True, scaled_to=.5, clipped=False, finalize=plot_mono_arr):
        osc1 = make_oscil(330)
        osc2 = make_oscil(500)
        for i in range(44100):
            outa(i, oscil(osc1) + oscil(osc2))


This is an example of defining a reverb function that is applied after the initial output is
rendered. using locsig makes this even easier.

Note ClM.reverb gets defined within the with Sound context.

::

    def bad_mono_rev1(volume=1.):
        ap1 = allpass1 = make_all_pass(-.9, .9, 1051)
        ap2 = allpass2 = make_all_pass(-.9, .9, 1207)
        ap3 = allpass3 = make_all_pass(-.9, .9, 1000)
        dly = make_delay(seconds2samples(.011))
        length = clm_length(CLM.reverb)
    
        apb = make_all_pass_bank([ap1, ap2, ap3])
    
        for i in range(length):
            outa(i, delay(dly, volume * all_pass_bank(apb, ina(i, CLM.reverb))))

    with Sound('ex2.aiff', play=True, statistics=True, reverb=bad_mono_rev1):#, reverb_data={'volume' : .5}):
        osc = make_oscil(500)
        e = make_env([0.,0.0, .05, .8, .4, 0.0, 1.0, 0.0], length=22050)
        for i in range(44100*2):
            val = env(e)*oscil(osc)
            outa(i, val*.6)
            outa(i, val*.4, CLM.reverb)



Lastly and example that demonstrates defining functions that can be called in Sound context
that uses finalize to plot the output file after the reverb has been applied

::

    from matplotlib import pyplot as plt

    def plot_mono_soundfile(filename):
        y, _ = file2ndarray(filename)
        plt.figure(figsize=(6,6))
        plt.plot(y[0], color='gray')
        plt.tight_layout()
        plt.show()



    def bad_mono_rev1(volume=1.):
        ap1 = allpass1 = make_all_pass(-.9, .9, 1051)
        ap2 = allpass2 = make_all_pass(-.9, .9, 1207)
        ap3 = allpass3 = make_all_pass(-.9, .9, 1000)
        dly = make_delay(seconds2samples(.011))
        length = clm_length(CLM.reverb)
    
        apb = make_all_pass_bank([ap1, ap2, ap3])
    
        for i in range(length):
            outa(i, delay(dly, volume * all_pass_bank(apb, ina(i, CLM.reverb))))
        
        
    def blip(start, dur, freq):
        osc = make_oscil(freq)
        beg = seconds2samples(start)
        end = beg + seconds2samples(dur)
        e = make_env([0.,0.0, .05, .8, .4, 0.0, 1.0, 0.0], duration=dur*.5)
        for i in range(beg, end):
            val = oscil(osc) * env(e)
            outa(i, val*.6)
            outa(i, val*.4, CLM.reverb)


    with Sound('ex2.aiff', play=True, statistics=True, reverb=bad_mono_rev1, finalize=plot_mono_soundfile):
        blip(0, 1, 400)
        blip(1, 1, 500)
        blip(2, 1, 600)
        blip(3, 1, 900)
        for i in np.arange(4, 6, .333333):
            blip(i, .5, 800)




