import os
os.system('cmd /c "git checkout master"')
os.system('cmd /c "git status"')
os.system('cmd /c "git add GBOOST"')
os.system('cmd /c "git add Prophet"')
os.system('cmd /c "git add RandomForest"')
os.system('cmd /c "git add Lasso"')
os.system('cmd /c "git add Arima"')
os.system('cmd /c "git commit -m "Update Models in git REPO""')
