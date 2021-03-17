package transforms;

import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

import java.util.*;
import java.lang.Math;

public class WrapTryCatch extends AverlocTransformer {
//   protected int UID = 0;

  public WrapTryCatch(int uid, Boolean allSites) {
    this.setUID(uid);
    this.setAllSites(allSites);
  }

	@Override
	public int transform(CtExecutable method, int... i) {

    this.setUID(i[0]);
    HashMap<String, List<String>> sites = new HashMap<String, List<String>>();

    ArrayList<CtLiteral> literals = getChildrenOfType(
      method, CtLiteral.class
    );

    CtTry wrapper = getFactory().Core().createTry();

    wrapper.addCatcher(
      getFactory().Code().createCtCatch(
        "REPLACEME" + Integer.toString(this.UID),
        java.lang.Exception.class,
        getFactory().Code().createCtBlock(
          getFactory().Code().createCtThrow(
            "REPLACEME" + Integer.toString(this.UID)
          )
        )
      )
    );

    sites.put(String.format("@R_%s@", Integer.toString(this.UID)), Arrays.asList("", this.getOutName()));

    wrapper.setBody(method.getBody());

    method.setBody(
      getFactory().Code().createCtBlock(
        wrapper
      )
    );
    this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);

    this.setChanged(method);
    return this.UID+1;
	}
}
