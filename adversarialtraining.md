Adversarial training with image, which is somewhat easier than with NLP 


https://stackoverflow.com/questions/52292861/fast-gradient-sign-method-with-keras


    def customLoss(x):
        def neg_log_likelihood(y_true, y_pred):

            def neg_log(y_t,y_p):
                inter=(y_p[...,0,None]-y_t)/K.clip(y_p[...,1,None],K.epsilon(),None)
                val=K.log(K.clip(K.square(y_p[...,1,None]),K.epsilon(),None))+K.square(inter)
                return val

            val=neg_log(y_true,y_pred)

            deriv=K.gradients(val,x)
            xb=x+0.01*K.sign(deriv)
            out=model(xb)
            valb=neg_log(y_true,out)

            return K.mean(val+valb,axis=-1)
        return neg_log_likelihood

    
