public boolean isMatchMyOwn(String s, String p, int currentposS, int currentposP) {        
        if (currentposS == s.length() && currentposP == p.length()) {
            return true;
        }
        if(currentposS>s.length())
            return false;
        if(currentposP>p.length())
            return false;
        char c1 = ' ';
        if (currentposS < s.length()) {
            c1 = s.charAt(currentposS);
        }
        char c2 = '@';
        if (currentposP < p.length()) {
            c2 = p.charAt(currentposP);
        }
        
        if (c1 == c2 || c2 == '?') {
            return isMatchMyOwn(s, p, currentposS + 1, currentposP + 1);
        }
        if (c2 == '*') {
            boolean a = false, b = false, c = false;
            
            if (currentposP + 1 < p.length()) {
                if (p.charAt(currentposP + 1) == '*') {
                    return isMatchMyOwn(s, p, currentposS, currentposP + 1);
                }
                if (c1 != ' ') {
                    a = isMatchMyOwn(s, p, currentposS + 1, currentposP + 1);
                    if(a==true)
                        return a;
                }
            }
            if (c1 != ' ') {
                c = isMatchMyOwn(s, p, currentposS + 1, currentposP);
                if(c==true)
                    return c;
            }
            b = isMatchMyOwn(s, p, currentposS, currentposP + 1);
            return b;
        }
        
        return false;
    }
    
    
    //the most elegant solution is here: http://yucoding.blogspot.com/2013/02/leetcode-question-123-wildcard-matching.html
