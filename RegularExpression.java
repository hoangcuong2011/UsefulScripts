public boolean isMatchMyOwn(String s, String p, int currentposS, int currentposP) {
        if (currentposS == s.length() && currentposP == p.length()) {
            return true;
        }
        if(currentposS>s.length())
            return false;
        char c1 = ' ';
        if (currentposS < s.length()) {
            c1 = s.charAt(currentposS);
        }
        char c2 = '@';
        if (currentposP < p.length()) {
            c2 = p.charAt(currentposP);
        }
        
        if (c1 == c2 || c2 == '.') {
            if (currentposP + 1 < p.length()) {
                if (p.charAt(currentposP + 1) == '*') {
                    return isMatchMyOwn(s, p, currentposS + 1, currentposP + 2)
                            || isMatchMyOwn(s, p, currentposS, currentposP + 2)
                            || isMatchMyOwn(s, p, currentposS + 1, currentposP);
                }
            }
            return isMatchMyOwn(s, p, currentposS + 1, currentposP + 1);
        }
        
        if (c1 != c2) {
            if (currentposP + 1 < p.length()) {
                if (p.charAt(currentposP + 1) == '*') {
                    return isMatchMyOwn(s, p, currentposS, currentposP + 2);

                }
            }
        }
        
        return false;
    }
