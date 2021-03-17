package transforms;

import java.util.*;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

public class RenameParameters extends Renamer<CtParameter> {
    public RenameParameters(int uid, Boolean allSites) {
        this.setUID(uid);
        this.setAllSites(allSites);
    }

	@Override
	public int transform(CtExecutable method, int... i) {
        // Reset prior to next transform
        reset();
	    this.setUID(i[0]);
        
        // Get setup for renaming
        setDefs(getChildrenOfType(method, CtParameter.class));
        int res; 
        if (this.allSites) {
            res = takePercentage(1);
        } else {
            res = takeSingle();
        }

        // Build new names and apply them
        applyTargetedRenaming(method, false);
	return res+this.UID;
	}
}