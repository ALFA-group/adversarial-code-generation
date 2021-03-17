package transforms;

import spoon.processing.AbstractProcessor;

import spoon.reflect.cu.*;
import spoon.reflect.cu.position.*;
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

public class Renamer<T extends CtNamedElement> extends AverlocTransformer {
  protected boolean debug = false;

//   @Override
//   public int UID;

  protected ArrayList<T> theDefs;
  protected ArrayList<T> targetDefs;

  protected ArrayList<String> subtokenBank;
  protected ArrayList<String> namesOfDefs;
  protected ArrayList<String> namesOfTargetDefs;

  protected void reset() {
    theDefs = new ArrayList<T>();
    targetDefs = new ArrayList<T>();
    namesOfDefs = new ArrayList<String>();
    namesOfTargetDefs = new ArrayList<String>();
    // this.siteMap = new HashMap<String, HashMap>();
  }

//   @Override
//   public void setUID(int uid) {
//     this.UID = uid;
//   }

  protected void setDefs(ArrayList<T> defs) {
    theDefs = new ArrayList<T>(defs);
    namesOfDefs = new ArrayList<String>();

    for (T def : defs) {
      namesOfDefs.add(def.getSimpleName());
    }

    if (debug) {
      System.out.println(String.format(
        "[RENAMER] - Recieved %s defs.", defs.size()
      ));
      System.out.println(String.format(
        "[RENAMER] - Recieved %s def names.", namesOfDefs.size()
      ));
    }
  }

  protected void setSubtokens(ArrayList<String> subtokens) {
    subtokenBank = subtokens;

    if (debug && subtokens != null) {
      System.out.println(String.format(
        "[RENAMER] - Recieved a corpus of %s frequent subtokens for random name building.",
        subtokenBank.size()
      ));
    }
  }

  protected int takeSingle() {
    Collections.shuffle(theDefs);

    int toTake = theDefs.size() > 0 ? 1 : 0;

    if (toTake <= 0) {
      if (debug) {
        System.out.println("[RENAMER] - Note: zero defs selected for transform.");
      }
      return 0;
    }

    targetDefs = new ArrayList<T>(theDefs.subList(0, 1));
    namesOfTargetDefs = new ArrayList<String>();

    for (T targetDef : targetDefs) {
      namesOfTargetDefs.add(targetDef.getSimpleName());
    }

    if (debug) {
      System.out.println(String.format(
        "[RENAMER] - Selected %s defs.", toTake
      ));
    }
    return targetDefs.size();
  }

  protected int takePercentage(double percentage) {
    Collections.shuffle(theDefs);

    int toTake = (int)Math.floor(theDefs.size() * percentage);

    if (toTake <= 0) {
      if (debug) {
        System.out.println("[RENAMER] - Note: zero defs selected for transform.");
      }
      return 0;
    }

    targetDefs = new ArrayList<T>(theDefs.subList(0, toTake));
    namesOfTargetDefs = new ArrayList<String>();

    for (T targetDef : targetDefs) {
      namesOfTargetDefs.add(targetDef.getSimpleName());
    }

    if (debug) {
      System.out.println(String.format(
        "[RENAMER] - Selecting %s%% (%s) of defs.", percentage*100.0, toTake
      ));
    }
    return targetDefs.size();
  }

  protected IdentityHashMap<T, String> generateRenaming(CtExecutable method, boolean shuffle, int nameMinSubtokens, int nameMaxSubtokens) {
    IdentityHashMap<T, String> renames = new IdentityHashMap<T, String>();
    
    String methodName = ((CtTypeMember)method).getDeclaringType().getSimpleName().toLowerCase();

    if (targetDefs == null || targetDefs.size() <= 0) {
      if (debug) {
        System.out.println("[RENAMER] - No renaming to be done: no targets selected.");
      }
      return null;
    }

    if (debug) {
      if (shuffle) {
        System.out.println("[RENAMER] - Generating renaming: mode=SHUFFLE.");
      } else {
        System.out.println(String.format(
          "[RENAMER] - Generating renaming: mode=RANDOM; nameMinSubtokens=%s; nameMaxSubtokens=%s.",
          nameMinSubtokens, nameMaxSubtokens
        ));
      }
    }

    if (shuffle) {
      if (targetDefs.size() <= 1) {
        if (debug) {
          System.out.println(String.format(
            "[RENAMER] - Cannot use shuffle mode with %s targets.", targetDefs.size()
          ));
        }
        return null;
      }

      Random rshuf = new Random();

      // Keep shuffling till we don't assign any name to itself
      // This generates a derangement but it's slow (~3x compared to 
      // just a permutation in this implementation) so maybe let's 
      // use some tricks to get (non-uniform) random (near) derangements?
      // boolean validShuffle = false;
      // while (!validShuffle) {
        // validShuffle = true;
      Collections.shuffle(namesOfTargetDefs);
      for (int i = 0; i < targetDefs.size(); i++) {
          if (namesOfTargetDefs.get(i).equals(targetDefs.get(i).getSimpleName())) {
              Collections.swap(
                namesOfTargetDefs, 
                i,
                rshuf.nextInt(namesOfTargetDefs.size() - 1)
              );
          }
      }
      // }

      // Setup those renames now that we have a good shuffle
      for (int i = 0; i < targetDefs.size(); i++) {
        renames.put(targetDefs.get(i), namesOfTargetDefs.get(i));
      }
    } else {
      Random rand = new Random();
      ArrayList<String> newNames = new ArrayList<String>();

      for (T target : targetDefs) {
        Collections.shuffle(subtokenBank);

        int length = 0;
        String name = null;
        do {
          length = rand.nextInt((nameMaxSubtokens - nameMinSubtokens) + 1) + nameMinSubtokens;
          name = camelCased(
            subtokenBank.stream().filter(
              // Make sure that we are not getting a subtoken in the method name, or any of the other names
              subtok -> !methodName.contains(subtok) && namesOfDefs.stream().allMatch(
                otherName -> !otherName.toLowerCase().contains(subtok)
              )
            ).collect(Collectors.toList()).subList(0, length)
          );
        } while (namesOfDefs.contains(name) || newNames.contains(name)); // Make sure we don't clash with pre-existing/new name

        newNames.add(name);
        renames.put(target, name);
      }
    }

    if (debug) {
      System.out.println("[RENAMER] - Generated renaming:");
      for (Map.Entry<T,String> item : renames.entrySet()) {
        System.out.println(String.format(
          "[RENAMER]   + Renamed: %s ==> %s.", item.getKey().getSimpleName(), item.getValue()
        ));
      }
    }

    return renames;
  }

  protected String safeGetLine(CtElement element) {
    SourcePosition position = element.getPosition();

    if (position instanceof NoSourcePosition) {
      return "???";
    } else {
      return String.format("%s", position.getLine());
    }
  }

  protected void applyTargetedRenaming(CtExecutable method, boolean skipDecls) {
    IdentityHashMap<T, String> renames = new IdentityHashMap<T, String>();
    HashMap<String, List<String>> sites = new HashMap<String, List<String>>();

    int i = this.UID;
    for (T target : targetDefs) {
      renames.put(target, "REPLACEME" + Integer.toString(i));
      // sites.put(String.format("@R_%s@", i), target.getSimpleName());
      sites.put(String.format("@R_%s@", i), Arrays.asList(target.getSimpleName(), this.getOutName()));
      i += 1;
    }

    applyRenaming(method, skipDecls, renames);
    this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);
  }

  protected void applyRenaming(CtExecutable method, boolean skipDecls, IdentityHashMap<T, String> renames) {
    // if (((CtTypeMember)method).getDeclaringType().getSimpleName().equals("WRAPPER_0793faf53642d8a7d6de593630f9eea1137ba84873641ea14ac1c1eee1314971")) {
    //   debug = true;
    // } else {
    //   debug = false;
    // }

    ArrayList<CtVariableAccess> usages = getChildrenOfType(method, CtVariableAccess.class);

    if (debug) {
      System.out.println("[RENAMER] - Applying renaming:");
    }

    if (renames == null) {
      if (debug) {
        System.out.println("[RENAMER]   + No renames provided: stopping.");
      }
      return;
    }

    for (CtVariableAccess usage : usages) {
      CtVariableReference<?> reference = usage.getVariable();
      if (reference != null) {
        CtNamedElement theDecl = reference.getDeclaration();
        if (renames.containsKey(theDecl)) {
          if (debug) {
            System.out.println(String.format(
              "[RENAMER]   + Applied renaming (%s ==> %s) to ref at line %s.",
              theDecl.getSimpleName(), renames.get(theDecl), safeGetLine(usage)
            ));
          }

          reference.setSimpleName(renames.get(theDecl));
          this.setChanged(method);
        }
      }
    }

    if (skipDecls) {
      if (debug) {
        System.out.println(String.format(
          "[RENAMER]   + Skipping decls."
        ));
      }
      return;
    }

    for (Map.Entry<T,String> item : renames.entrySet()) {
      if (debug) {
        System.out.println(String.format(
          "[RENAMER]   + Applied renaming (%s ==> %s) to decl at line %s.",
          item.getKey().getSimpleName(), item.getValue(), safeGetLine(item.getKey())
        ));
      }

      item.getKey().setSimpleName(item.getValue());
      this.setChanged(method);
    }
  }

}