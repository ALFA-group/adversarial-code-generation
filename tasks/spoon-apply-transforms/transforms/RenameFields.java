package transforms;

import spoon.reflect.factory.*;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

import java.util.*;
import java.util.stream.*;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.IOException;
import java.lang.Math;

public class RenameFields extends Renamer<CtField> {
  public RenameFields(int uid, Boolean allSites) {
    this.setUID(uid);
    this.setAllSites(allSites);
  }

	@Override
	public int transform(CtExecutable method, int... i) {
    // Reset prior to next transform
    reset();

    this.setUID(i[0]);

    // Find field references with no declaring type. These should
    // be fields of the enclosing class but, since we just have a method
    // we've lost the enclosing class
    ArrayList<CtFieldReference> fieldReferences = new ArrayList<CtFieldReference>(
      getChildrenOfType(
        method, CtFieldReference.class
      ).stream().filter(
        // Only want ones from our class
        fieldRef -> fieldRef.getDeclaringType() == null
      ).collect(
        Collectors.toList()
      )
    );

    // First, collect unique field names
    ArrayList<String> fieldNames = new ArrayList<String>();
    for (CtFieldReference fieldRef : fieldReferences) {
      if (!fieldNames.contains(fieldRef.getSimpleName())) {
        fieldNames.add(fieldRef.getSimpleName());
      }
    }

    // Then, we'll build the field declarations (virtually)
    // and attach them to our fake class (and save them for our
    // renamer to use later)
    ArrayList<CtField> fieldDecls = new ArrayList<CtField>();
    for (String fieldName : fieldNames) {
      CtField<String> generatedField = ((CtTypeMember)method).getDeclaringType().getFactory().Field().create(
        null,
        new HashSet<>(),
        ((CtTypeMember)method).getDeclaringType().getFactory().Type().STRING,
        fieldName
      );

      ((CtTypeMember)method).getDeclaringType().addField(generatedField);
      fieldDecls.add(generatedField);
    }

    // Finally, we'll go ahead and link our references to the faked
    // declarations (thus, placing us in a state where we can use our
    // renamer logic)
    for (CtFieldReference fieldRef : fieldReferences) {
      for (CtField<String> fieldDecl : fieldDecls) {
        if (fieldDecl.getSimpleName().equals(fieldRef.getSimpleName())) {
          fieldRef.setDeclaringType(
            ((CtTypeMember)method).getDeclaringType().getFactory().Type().createReference(
              fieldDecl.getDeclaringType()
            )
          );
        }
      }
    }

    // Get setup for renaming (use the fieldDecls we crafted earlier)
    setDefs(fieldDecls);
    
    // Select fields to rename
    int res;
    if (this.allSites) {
        res = takePercentage(1);
    } else {
        res = takeSingle();
    }

    // Build new names and apply them (true ==> to skip decls)
    applyTargetedRenaming(method, true);

    // Cleanup: remove fields from WRAPPER class
    for (CtField<String> generatedField : fieldDecls) {
      ((CtTypeMember)method).getDeclaringType().removeField(generatedField);
    }

    return res+this.UID;
	}
}
