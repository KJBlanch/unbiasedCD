Restricted Boltzmann Machine, Julia Package. 

For use, we recommend running a Pluto notebook locally. The material in this instructional is based on "Computational Thinking, a live online Julia/Pluto textbook, https://computationalthinking.mit.edu"

Step 1:
  Go to https://julialang.org/downloads and download the current stable release for your operating system.

Step 2:
  After installing, verify that your version of Julia is working. Each operating system is slightly different, but you should be able to search for Julia in your applications to open a Julia terminal, or type 'julia' in an open terminal to start the interface. 
  Once open, you can verify the installation by simply using the command '1 + 1', and you should the correct execution.
  
Step 3:
  Install Pluto. Do this by using the command ']'. You should see the line turn blue, and the prompt change to 'pkg', indicating that it is in package mode. 
  Start the installation by executing 'add Pluto'. This is only required once per installation of Julia. 
  At this point, MIT suggest you go enjoy a nice hot beverage whilst you wait for the installation, and we second that suggestion. 
  
Step 4:
  You will need a modern browser to interact and view the Pluto notebook, and so for this, Firefox or Chrome is recommended. 
  
Step 5:
  Open up a new terminal (like step 2) and execute the following:
  'using Pluto'
  'Pluto.run()'
  
  You should now see localhost address that you can open in your browser (if it hasn't automatically done so), and from there, you can open the .jl book included in this repository by downloading it locally and opening the file location. Enjoy!
