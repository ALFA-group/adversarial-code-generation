package transforms;

import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

import java.util.*;
import java.lang.Math;

public class ReplaceTrueFalse extends AverlocTransformer {
//   protected int UID = 0;

  public ReplaceTrueFalse(int uid, Boolean allSites) {
    this.setUID(uid);
    this.setAllSites(allSites);
  }

  public void replaceLiteral(CtLiteral target, int siteID) {
    CtBinaryOperator<Boolean> replacement = getFactory().Code().createBinaryOperator(
        getFactory().Code().createLiteral("REPLACEME" + Integer.toString(siteID)), 
        getFactory().Code().createLiteral("REPLACEME" + Integer.toString(siteID)), 
        ((Boolean)target.getValue()) ? BinaryOperatorKind.EQ : BinaryOperatorKind.NE
      );
  
      target.replace(replacement);
  }

	@Override
	public int transform(CtExecutable method, int... i) {
    
    this.setUID(i[0]);

    HashMap<String, List<String>> sites = new HashMap<String, List<String>>();

    ArrayList<CtLiteral> literals = getChildrenOfType(
      method, CtLiteral.class
    );
    
    
    literals.removeIf(
    x -> x.getValue() == null || !(x.getValue() instanceof Boolean)
    );
    

    if (literals.size() <= 0) {
      this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);
      return this.UID;
    }

    int res;


    if (this.allSites) {
        for (int j=0; j<literals.size(); j++) {
            CtLiteral target = literals.get(j);
            replaceLiteral(target, j+this.UID);
            if ((Boolean)target.getValue()) {
                sites.put(String.format("@R_%s@ %s @R_%s@", Integer.toString(this.UID+j), "==", Integer.toString(this.UID+j)), Arrays.asList("true", this.getOutName()));
            } else {
                sites.put(String.format("@R_%s@ %s @R_%s@", Integer.toString(this.UID+j), "!=", Integer.toString(this.UID+j)), Arrays.asList("false", this.getOutName()));
            }
        }
        res = literals.size()+this.UID;
    } else {
        Collections.shuffle(literals);
        CtLiteral target = literals.get(0);
        replaceLiteral(target, this.UID);
        if ((Boolean)target.getValue()) {
            sites.put(String.format("@R_%s@ %s @R_%s@", Integer.toString(this.UID), "==", Integer.toString(this.UID)), Arrays.asList("true", this.getOutName()));
        } else {
            sites.put(String.format("@R_%s@ %s @R_%s@", Integer.toString(this.UID), "!=", Integer.toString(this.UID)), Arrays.asList("false", this.getOutName()));
        }
        res = this.UID+1;
    }

    
    this.setChanged(method);
    this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);
    return res;
	}
}
