<h1 style="border: 0;"> Neural Machine Translated communication system </h1>


<img src="https://github.com/Nix-code/Neural-Machine-Translated-communication-system/blob/main/public/model-assets/demo.gif" height="450">

The model is basically direct to convert one source language to another targeted language using encoder and decoder architecture. The model encodes the message sent by the sender to a vector of fixed length and decoder generates the translated message which is received by the receiver in their communication system(chat application) automatically.

### Project status (Pending work)
- Backend improvements yet to be done
- User authentication and UI improvement 

# Table Of Contents

-   [Prerequisites](#prerequisites)
-   [Contribute](#Contribute)
-   [About](#About)
-   [Logic](#Logic)
-   [Evaluation](#Evaluation)
-   [Licence](#Licence)



## Prerequisites

-   Install python packages such as `numpy` `pandas` `Tensorflow` `Django` `matplotlib`


## Contribute


-   Fork the repository
-   Commit your changes
-   create Pull request

## About
The model is trained using the spanish-english dataset with 100 epochs. The dataset contains about 110k rows and took about 4 hours to train using Nvidia GTX 1650 graphics card.
### Why not using Google API for language translator?
<p> Not used Google translator API beacase, I wanted to make ML model from scratch without using any API's. I believe, using API(In this case), is very good way to make the translator but I never did this project in the sake of just making it work rather than learning from it. Making end to end attention model from scratch helped to learn about how actually the neural machine translation work.</p>

## Logic
- Logic behind sender and receiver's communication system.
<img src="https://github.com/Nix-code/Neural-Machine-Translated-communication-system/blob/main/public/model-assets/logic_send_receive.png">

## Evaluation
```
Epoch 100 Batch 600 Loss 0.24747854098677635
Epoch 100 Loss 0.0356
Time taken for 1 epoch 174.43703937530518 sec
```

## Clone the project

```
git clone git@github.com:Nix-code/Nix-code-Neural-Machine-Translation-communication-system-.git
```

## Run Django web application in local host
```
python3 manage.py runserver
```
or
```
bash run.sh
```


## Licence
```MIT```
