package transforms;

import java.util.*;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

public class Sequenced extends AverlocTransformer {
  ArrayList<AverlocTransformer> subTransformers;

  public Sequenced(ArrayList<AverlocTransformer> subTransformers, String name, int uid) {
    this.subTransformers = subTransformers;
    this.name = name;
    this.setUID(uid);
  }

	@Override
	public int transform(CtExecutable method, int... j) {
    int i = j[0];
    HashMap<String, List<String>> sites = new HashMap<String, List<String>>();
    String methodName = ((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", "");
    this.siteMap.put(methodName, sites);
    for (AverlocTransformer transformer : subTransformers) {
      transformer.setFactory(getFactory());
      i = transformer.transform(method, i);
      if (transformer.changes(((CtTypeMember)method).getDeclaringType().getSimpleName())) {
        this.setChanged(method);
      }
      this.siteMap.get(methodName).putAll(transformer.siteMap.get(methodName));
    }
    return i;
	}
}