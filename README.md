# final
這次期末專案我用了Lobe建立一個影像辨識的模型  

功能是可以辨識被攝影機照到的人有沒有戴口罩

Lobe是微軟推出的離線版神經網路分類工具  
(目前能用於圖像的分類，之後才會有比較多的功能)  
相較於其他類似的工具，Lobe比較大的優勢就是能離線操作  
可以在官網下在好之後進入新增一個專案  
<img src="https://github.com/Sred0626/final/blob/main/%E9%96%8B%E5%A7%8B.png" width="900" lenth="900" />

右上角的import可以選擇要匯入的資料(圖片)  

<img src="https://github.com/Sred0626/final/blob/main/%E5%8C%AF%E5%85%A5.png" width="500" lenth="500"/>


匯入的方式有3種  
1.匯入單一圖片  
2.馬上用相機拍照並且匯入  
3.匯入一個資料夾(裡面的圖片需要是分類好的狀態)  


因為這次我是做判斷人有沒有戴口罩的功能，所以我的資料放入的圖片就分成兩種lable  
一種是有戴口罩(lable=>mask),另一種沒有戴口罩(lable=>notmask)  
匯入完成並且標記好圖片lable之後就會自動開始進行模型學習的部分  

等待模型訓練好之後就可以匯出了  
這裡匯出的方式有很多種，而我選擇的是tenserflow  

<img src="https://github.com/Sred0626/final/blob/main/%E5%8C%AF%E5%87%BA.png" width="500" lenth="500"/>

匯出完以後會得到一個以(你在Lobe專案的名字 TensorFlow)為名字的資料夾  
資料夾裡有一個tf_example.py的檔案  
這個檔案打開來看會發現他已經把如何引用我剛剛訓練好的模型，初始化，預測結果的功能都寫好了  
我們只需要想要如何獲取他預測好的結果以及用什麼方法呈現  

下面是測試的結果  

<img src="https://github.com/Sred0626/final/blob/main/test1.png" width="500" lenth="500"/>  
<img src="https://github.com/Sred0626/final/blob/main/test2.png" width="500" lenth="500"/>  
<img src="https://github.com/Sred0626/final/blob/main/test3.png" width="500" lenth="500"/>
