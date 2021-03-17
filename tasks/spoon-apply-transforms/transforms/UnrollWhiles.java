package transforms;

import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

import java.util.*;
import java.lang.Math;

public class UnrollWhiles extends AverlocTransformer {
  protected int UNROLL_STEPS = 1;
//   protected int UID = 0;

  public UnrollWhiles(int uid, Boolean allSites) {
    this.setUID(uid);
    this.setAllSites(allSites);
  }

  public void processWhiles(CtWhile target) {
    CtStatement whileBody = target.getBody();
    CtStatement lastBody = target;
    for (int i = 0; i < this.UNROLL_STEPS; i++) {
        CtWhile wrapperIf = getFactory().Core().createWhile();
  
        wrapperIf.setLoopingExpression(target.getLoopingExpression().clone());
  
        CtBlock temp = getFactory().Core().createBlock();
        temp.addStatement(whileBody.clone());
        temp.addStatement(lastBody.clone());
        temp.addStatement(getFactory().Core().createBreak());
  
        wrapperIf.setBody(temp);
  
        lastBody = wrapperIf.clone();
      }
  
      target.replace(lastBody);

  }

	@Override
	public int transform(CtExecutable method, int... i) {
        
    this.setUID(i[0]);

    ArrayList<CtWhile> whiles = getChildrenOfType(
      method, CtWhile.class
    );
    
    whiles.removeIf(
      x -> x.getBody() == null || x.getLoopingExpression() == null
    );

    HashMap<String, List<String>> sites = new HashMap<String, List<String>>();
    this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);

    if (whiles.size() <= 0) {
      return this.UID;
    }

    if (this.allSites) {
        for (CtWhile target : whiles) {
            processWhiles(target);
        }
    } else {
        Collections.shuffle(whiles);
        CtWhile target = whiles.get(0);
        processWhiles(target);
    }

    
    this.setChanged(method);
    return this.UID;
	}
}
