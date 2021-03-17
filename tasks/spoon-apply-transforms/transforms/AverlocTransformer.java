package transforms;

import spoon.processing.AbstractProcessor;

import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

import spoon.reflect.visitor.CtIterator;

import java.util.*;
import java.util.stream.*;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.IOException;
import java.lang.Math;

public class AverlocTransformer extends AbstractProcessor<CtExecutable> {
  protected ArrayList<String> topTargetSubtokens;

  protected ArrayList<String> changes = new ArrayList<String>();

  protected String name = null;

  public HashMap<String, HashMap> siteMap = new HashMap<String, HashMap>();

  protected Boolean allSites;

  protected int UID;

  public void setUID(int uid) {
    this.UID = uid;
  }

  public String getOutName() {
    if (name != null) {
      return name;
    }
    return this.getClass().getName();
  }

  public boolean changes(String name) {
    return this.changes.contains(name);
  }

  protected void setAllSites(Boolean allSites) {
    this.allSites = allSites;
  }

  protected void setChanged(CtExecutable method) {
    String name = ((CtTypeMember)method).getDeclaringType().getSimpleName();
    if (this.changes.contains(name)) {
      return;
    }
    changes.add(name);
  }
  
  public void setTopTargetSubtokens(ArrayList<String> topTargetSubtokens) {
    this.topTargetSubtokens = topTargetSubtokens;
  }

  protected <T extends CtElement> ArrayList<T> getChildrenOfType(CtExecutable method, Class<T> baseCls) {
    ArrayList<T> results = new ArrayList<T>();
    CtIterator iter = new CtIterator(method);
    while (iter.hasNext()) {
        CtElement el = iter.next();
        if (baseCls.isInstance(el)) {
          results.add((T)el);
        }
    }
    return results;
  }

  // Build a camel cased name from a randomized sub-token list
  protected String camelCased(List<String> inputs) {
      String retVal = inputs.get(0);
      for (String part : inputs.subList(1, inputs.size())) {
          retVal += part.substring(0, 1).toUpperCase() + part.substring(1);
      }
      return retVal;
  }

	@Override
	public void process(CtExecutable element) {
    // Skip this 
    if (element.getSimpleName().equals("<init>")) {
      return;
    }

    // Skip lambda impls
    if (!(element instanceof CtTypeMember)) {
      return;
    }
    
    // Also skip nested things...
    if (!((CtTypeMember)element).getDeclaringType().getSimpleName().startsWith("WRAPPER_")) {
      return;
    }

    // Also skip things with null body
    if (element.getBody() == null) {
      return;
    }

    transform(element, this.UID);
  }

  protected void transform(CtExecutable method) {

  }
  protected int transform(CtExecutable method, int... i){
  return i[0];
  }
}