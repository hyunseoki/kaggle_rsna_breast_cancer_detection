function PyScript{
    param([String]$cmd)
    $host.UI.RawUI.WindowTitle = $cmd
    Invoke-Expression -Command $cmd
}

$device = "cuda:0"


################################################################################################
$cmd = "python ./src/main.py" +
        " --device $device " +
        " --epochs 10 " +
        " --loss_weight 20" +
        " --loss_type bce " +
        " --use_wandb True " +
        " --train_oversample True "
PyScript($cmd)

################################################################################################
$cmd = "python ./src/main.py" +
        " --device $device " +
        " --epochs 10 " +
        " --loss_weight 20" +
        " --loss_type bce " +
        " --use_wandb True " +
        " --train_oversample False "
PyScript($cmd)

################################################################################################
$cmd = "python ./src/main.py" +
        " --device $device " +
        " --epochs 10 " +
        " --loss_type focal " +
        " --use_wandb True " +
        " --train_oversample False "
PyScript($cmd)

################################################################################################
$cmd = "python ./src/main.py" +
        " --device $device " +
        " --epochs 10 " +
        " --loss_type focal " +
        " --use_wandb True " +
        " --train_oversample True "
PyScript($cmd)