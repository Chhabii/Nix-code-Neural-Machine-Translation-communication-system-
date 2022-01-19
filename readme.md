<h1 style="border: 0;"> Neural Machine Translated communication system </h1>


<img src="https://raw.githubusercontent.com/Nix-code/Nix-code-Neural-Machine-Translation-communication-system-/main/src/nmt_chat/assets/static/images/nmt-chat.gif" height="450">

The model is basically direct to convert one source language to another targeted language using encoder and decoder architecture. The model encodes the message sent by the sender to a vector of fixed length and decoder generates the translated message which is received by the receiver in their communication system(chat application) automatically.

### Project status
- Ongoing 
- Backend improvements yet to be done

# Table Of Contents

-   [Prerequisites](#prerequisites)
-   [Contribute](#Contribute)
-   [About](#About)
-   [Evaluation](#Evaluation)
-   [Licence](#Licence)



## Prerequisites

-   Install python packages such as `numpy` `pandas` `Tensorflow` `Django` `matplotlib`


## Contribute


-   Fork the repository
-   Commit your changes
-   push to the branch & open a pull request

## About
The model is trained using the spanish-english dataset with 100 epochs. The dataset contains about 110k rows and took about 4 hours to train using Nvidia GTX 1650 graphics card.

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

## Licence
```MIT```
